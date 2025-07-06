import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
from dotenv import load_dotenv
import logging

# 加载环境变量
load_dotenv()

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY")
DOUBAO_API_BASE = os.getenv("DOUBAO_API_BASE", "https://ark.cn-beijing.volces.com/api/v3")
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL", "doubao-seed-1-6-flash-250615")

HEADERS = {
    "Authorization": f"Bearer {DOUBAO_API_KEY}",
    "Content-Type": "application/json"
}

@app.post("/v1/completions")
async def proxy_chat_completions(request: Request):
    try:
        # 解析请求体
        body = await request.json()
        # logger.info(f"Received request: {json.dumps(body, indent=2)}")
        
        # 提取参数并构建豆包API请求
        model = body.get("model", DOUBAO_MODEL)
        prompt = body.get("prompt", "")

        system_prompt = '''
You are an AI programming assistant. Your duty is to help users to modify their code.
The edit area (code need your modification) is surrounded by label <|CODE_START|><|CODE_END|>.
The request from user is surrounded by label <|REQ_START|><|REQ_END|>.
The code language is surrounded by label <|LANG_START|><|LANG_END|>.
Context is also provided.
Content before the edit area is surrounded by label <|PREFIX_BEGIN|><|PREFIX_END|>
Content after the edit area is surrounded by label <|SUFFIX_BEGIN|><|SUFFIX_END|>
Please modify the content of edit area and return the modified code directly without any extra label.
'''

        # 构建消息
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # 构建豆包API请求体
        chat_body = {
            "model": model,
            "messages": messages,
            "stream": True,  # 强制流式响应
            "max_tokens": body.get("max_tokens", 4096),
            # "temperature": body.get("temperature", 0.7),
            'thinking': {
                'type': 'auto'
            },
        }
        
        # 添加可选参数
        for param in ["top_p", "presence_penalty", "frequency_penalty"]:
            if param in body:
                chat_body[param] = body[param]
        
        # logger.info(f"Forwarding to Doubao: {json.dumps(chat_body, indent=2)}")
        
        # 创建异步HTTP客户端
        async with httpx.AsyncClient() as client:
            # 发送请求到豆包API
            response = await client.post(
                url=f"{DOUBAO_API_BASE}/chat/completions",
                headers=HEADERS,
                json=chat_body,
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 流式转发响应
            usage = None
            async def generate(resp):
                SSE_PREFIX = 'data: '
                SSE_END = 'data: [DONE]'
                LINE_DEMI = '\n\n'
                LINE_DEMI_BYTES = LINE_DEMI.encode('utf-8')
                new_chunk = b''
                first = True
                all_text = ''
                try:
                    # 直接流式转发响应内容
                    async for chunk in resp.aiter_bytes():
                        s = chunk.decode('utf-8').strip()
                        for elem in s.split('\n\n'):
                            if elem == SSE_END:
                                if first:
                                    first = False
                                else:
                                    new_chunk += LINE_DEMI_BYTES
                                new_chunk += elem.encode('utf-8')
                            else:
                                data = json.loads(s=elem.removeprefix(SSE_PREFIX))
                                choices = data['choices']
                                if len(choices) == 0:
                                    # but usage is not available for stream-response
                                    usage = data.get('usage')
                                    continue
                                text = choices[0]['delta']['content']
                                all_text += text
                                reasoning_text = choices[0]['delta'].get('reasoning_content', '')
                                if len(text) == 0 and len(reasoning_text) > 0:
                                    continue
                                if first:
                                    first = False
                                else:
                                    new_chunk += LINE_DEMI_BYTES
                                new_data = {
                                    'id': data['id'],
                                    'object': 'text_completion',
                                    'created': data['created'],
                                    'choices': [
                                        {
                                            'text': data['choices'][0]['delta']['content'],
                                            'index': data['choices'][0]['index'],
                                            'logprobs': None,
                                            'finish_reason': data['choices'][0].get('finish_reason'),
                                        }
                                    ],
                                    'model': data['model'],
                                }
                                new_chunk += (SSE_PREFIX + json.dumps(new_data)).encode('utf-8')

                        yield new_chunk
                except httpx.StreamClosed:
                    logger.info("Client closed connection")
                except Exception as e:
                    logger.error(f"Streaming error: {str(e)}")
                finally:
                    await response.aclose()
            
            # 返回流式响应
            return StreamingResponse(
                generate(resp=response),
                status_code=response.status_code,
                media_type="text/event-stream",
                headers=response.headers
            )
    
    except httpx.HTTPStatusError as e:
        logger.error(f"Doubao API error: {e.response.text}")
        return StreamingResponse(
            content=f"data: {json.dumps({'error': f'Doubao API error: {e.response.status_code}'})}\n\n",
            status_code=e.response.status_code,
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.exception(f"Internal server error: {str(e)}")
        return StreamingResponse(
            content=f"data: {json.dumps({'error': 'Internal server error'})}\n\n",
            status_code=500,
            media_type="text/event-stream"
        )