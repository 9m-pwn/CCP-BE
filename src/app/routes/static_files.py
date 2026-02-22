from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

SRC_ROOT = Path(__file__).resolve().parents[2]
HEATMAPS_ROOT = (SRC_ROOT / "logs" / "heatmaps").resolve()


@router.get("/static/heatmaps/{relative_path:path}")
async def get_heatmap_file(relative_path: str):
    requested = (HEATMAPS_ROOT / relative_path).resolve()

    if not str(requested).startswith(str(HEATMAPS_ROOT)):
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not requested.exists() or not requested.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=str(requested))
