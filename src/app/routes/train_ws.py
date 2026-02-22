import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import redis.asyncio as redis

from app.settings import get_redis_url

router = APIRouter()
REDIS_URL = get_redis_url()


async def get_training_status_from_redis():
    redis_client = redis.from_url(REDIS_URL)
    try:
        status = await redis_client.get("training_status")
        return json.loads(status) if status else {}
    except Exception as exc:
        return {
            "in_progress": False,
            "status": "redis_unavailable",
            "error": str(exc),
            "metrics": {},
        }
    finally:
        await redis_client.close()


@router.websocket("/ws/train-status")
async def websocket_train_status(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            status = await get_training_status_from_redis()
            await websocket.send_json(status)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
