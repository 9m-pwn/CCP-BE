# src/app/routes/train_ws.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import redis.asyncio as redis
import json

router = APIRouter()

REDIS_URL = "redis://localhost:6379"

async def get_training_status_from_redis():
    # Create a Redis client with decoding enabled
    redis_client = redis.from_url(REDIS_URL)
    status = await redis_client.get("training_status")
    await redis_client.close()
    return json.loads(status) if status else {}

@router.websocket("/ws/train-status")
async def websocket_train_status(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            status = await get_training_status_from_redis()
            await asyncio.sleep(1)
            await websocket.send_json(status)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
