# src/app/routes/predict.py

import os
from scipy.ndimage import zoom
import shutil
import tempfile
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.utils.hollow_circle_util import draw_hollow_circle_by_red_regions
import cv2
import numpy as np
from PIL import Image

from app.predict_module_res50 import (
    preprocess_image,
    predict_image,
    make_gradcam_heatmap,
    overlay_heatmap_on_image,
    overlay_hot_only
)

router = APIRouter(prefix="/predict", tags=["predict"])
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ROOT_PATH      = Path(__file__).parent.parent.parent
MODEL_PATH     = ROOT_PATH / "app" /"model_from_resnet50_cbam.keras"
VALID_EXTS     = {".jpg", ".jpeg", ".png"}


class PredictionResponse(BaseModel):
    predicted_label: str
    confidence: float


class HeatmapResponse(PredictionResponse):
    heatmap_path: str


def _validate_extension(fn: str):
    ext = Path(fn).suffix.lower()
    if ext not in VALID_EXTS:
        raise HTTPException(400, f"Invalid file extension: {ext}")
    return ext


def _save_upload(file: UploadFile, ext: str) -> Path:
    upload_dir = ROOT_PATH / "logs" / "heatmaps" / "predict"
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        prefix="upload_", suffix=ext, delete=False, dir=str(upload_dir)
    )
    shutil.copyfileobj(file.file, tmp)
    tmp.flush()
    file.file.close()
    return Path(tmp.name)


@router.post("/", response_model=PredictionResponse)
async def predict_only(file: UploadFile = File(...)):
    ext = _validate_extension(file.filename)
    tmp_path = _save_upload(file, ext)

    try:
        # ทำ inference ผ่านโมดูลตรง
        label, conf = predict_image(str(tmp_path), str(MODEL_PATH))
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return PredictionResponse(predicted_label=label, confidence=conf)


@router.post("/heatmap", response_model=HeatmapResponse)
async def predict_with_heatmap(file: UploadFile = File(...)):
    ext = _validate_extension(file.filename)
    tmp_path = _save_upload(file, ext)

    try:
        # 1) preprocess ทั้ง original RGB กับ tensor
        orig_img, inp_tensor = preprocess_image(str(tmp_path))

        # 2) predict label + confidence
        label, conf = predict_image(str(tmp_path), str(MODEL_PATH))

        # 3) สร้าง Grad-CAM heatmap
        model = predict_image.__globals__['_model_cache'][str(MODEL_PATH)]
        heatmap = make_gradcam_heatmap(model, inp_tensor)

        # 3.1) เตรียม pure heatmap สำหรับบันทึก
        #    1) ขยายให้เท่าภาพจริง
        h, w = heatmap.shape
        hm_resized = zoom(heatmap, (orig_img.shape[0]/h, orig_img.shape[1]/w))
        #    2) แปลงเป็น uint8
        heat_uint8 = np.uint8(255 * hm_resized)
        #    3) เอาไปลง colormap (หรือจะไม่ใช้ก็ได้ ถ้าต้องการ grayscale)
        pure_heatmap = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        #    4) save ไฟล์ pure heatmap
        pure_path = tmp_path.with_name(f"{tmp_path.stem}_pure_heatmap{ext}")
        cv2.imwrite(str(pure_path), pure_heatmap)

        # 4) overlay heatmap บนภาพ
        overlay_all = overlay_heatmap_on_image(orig_img, heatmap, alpha=0.4)
        heatmap_out_path = tmp_path.with_name(f"{tmp_path.stem}_gradcam{ext}")
        cv2.imwrite(
            str(heatmap_out_path), 
            cv2.cvtColor(overlay_all, cv2.COLOR_RGB2BGR)
        )

        # 5) วาดวงกลมรอบ hot zone
        circled = draw_hollow_circle_by_red_regions(orig_img, heatmap)
        circled_rgb = cv2.cvtColor(circled, cv2.COLOR_BGR2RGB)
        circled_path = tmp_path.with_name(f"{tmp_path.stem}_circled{ext}")
        cv2.imwrite(
            str(circled_path), 
            circled_rgb
        )

    except Exception as e:
        raise HTTPException(500, f"Heatmap error: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return HeatmapResponse(
        predicted_label=label, 
        confidence=conf, 
        heatmap_path=str(heatmap_out_path), 
    )
