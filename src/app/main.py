from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis

from app.routes.predict import router as predict_router
from app.routes.static_files import router as static_files_router
from app.routes.train import router as train_router
from app.routes.train_ws import router as train_ws_router
from app.settings import get_api_host, get_api_port, get_api_reload, get_redis_url


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis_client = redis.from_url(get_redis_url(), decode_responses=True)
    app.state.redis_connected = False
    try:
        try:
            pong = await app.state.redis_client.ping()
            app.state.redis_connected = bool(pong)
            if pong:
                print("Redis connection established:", pong)
        except Exception as exc:
            print(f"Redis unavailable at startup. Continuing without Redis. Reason: {exc}")
        yield
    finally:
        await app.state.redis_client.close()


app = FastAPI(title="Cervical Cancer Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, tags=["Predict"])
app.include_router(static_files_router, tags=["Static Files"])
app.include_router(train_router, prefix="/train", tags=["Train"])
app.include_router(train_ws_router, tags=["Train Status"])


@app.get("/health")
async def health():
    return {"status": "ok", "redis_connected": bool(app.state.redis_connected)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=get_api_host(),
        port=get_api_port(),
        reload=get_api_reload(),
    )
