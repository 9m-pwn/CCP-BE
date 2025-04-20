# src/app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.routes.predict import router as predict_router
from app.routes.train import router as train_router
from app.routes.train_ws import router as train_ws_router
import redis.asyncio as redis

REDIS_URL = "redis://localhost:6379"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: สร้าง Redis client และเก็บไว้ใน app.state
    # ในที่นี้เราใช้ redis.from_url เพื่อสร้าง connection ด้วยตัวเลือก decode_responses=True
    app.state.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    try:
        # ทดสอบการเชื่อมต่อ Redis ด้วยคำสั่ง ping
        pong = await app.state.redis_client.ping()
        if pong:
            print("Redis connection established:", pong)
        else:
            print("Failed to connect to Redis")
        yield
    finally:
        # Shutdown: ปิด Redis client เมื่อแอพหยุดทำงาน
        await app.state.redis_client.close()

app = FastAPI(title="Cervical Cancer Detection API", lifespan=lifespan)

# ตั้งค่า CORS policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# รวม routes ต่าง ๆ
app.include_router(predict_router, tags=["Predict"])
app.include_router(train_router, prefix="/train", tags=["Train"])
app.include_router(train_ws_router, tags=["Train Status"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
