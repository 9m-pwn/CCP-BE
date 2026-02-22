import uvicorn

from src.app.settings import get_api_host, get_api_port, get_api_reload


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        app_dir="src",
        host=get_api_host(),
        port=get_api_port(),
        reload=get_api_reload(),
    )
