import base64
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.predict_module_res50 import (
    build_gradcam_overlay,
    predict_image,
    preprocess_image,
)
from app.settings import get_predict_normal_threshold, get_predict_use_two_stage

router = APIRouter(prefix="/predict")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ROOT_PATH = Path(__file__).parent.parent.parent
MODEL_PATH = ROOT_PATH / "app" / "model_from_resnet50_cbam.keras"
VALID_EXTS = {".jpg", ".jpeg", ".png"}

LABEL_INFO = {
    "Normal": {
        "code": "NORMAL",
        "en": "No abnormal cervical lesion is detected.",
    },
    "LSIL": {
        "code": "LSIL",
        "en": "Low-grade squamous intraepithelial lesion (early, mild abnormality).",
    },
    "HSIL": {
        "code": "HSIL",
        "en": "High-grade squamous intraepithelial lesion (higher-risk precancerous change).",
    },
    "Invasive": {
        "code": "INVASIVE",
        "en": "Features suggest invasive cancer and require urgent specialist review.",
    },
}


def _default_label_meta(label: str) -> dict[str, str]:
    safe_code = (
        str(label)
        .upper()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(";", "_")
    )
    return {
        "code": safe_code[:64] or "UNKNOWN",
        "en": f"Model predicted class: {label}",
        "th": f"Model predicted class: {label}",
    }


class PredictResponse(BaseModel):
    predicted_label: str
    label_code: str
    label_description: str
    label_description_en: str
    label_description_th: str
    confidence: float
    heatmap_enabled: bool
    include_base64: bool
    original_image_path: str
    original_image_url: str
    display_image_path: str
    display_image_url: str
    heatmap_path: Optional[str] = None
    heatmap_url: Optional[str] = None
    original_image_base64: Optional[str] = None
    display_image_base64: Optional[str] = None
    heatmap_image_base64: Optional[str] = None


def _validate_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in VALID_EXTS:
        raise HTTPException(400, f"Invalid file extension: {ext}")
    return ext


def _save_upload(file: UploadFile, ext: str) -> Path:
    # Store uploads under logs/heatmaps/predict so static route can serve outputs.
    upload_dir = ROOT_PATH / "logs" / "heatmaps" / "predict"
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        prefix="upload_", suffix=ext, delete=False, dir=str(upload_dir)
    )
    shutil.copyfileobj(file.file, tmp)
    tmp.flush()
    file.file.close()
    return Path(tmp.name)


def _to_public_heatmap_url(path: Path) -> str:
    heatmaps_root = ROOT_PATH / "logs" / "heatmaps"
    relative = path.relative_to(heatmaps_root).as_posix()
    return f"/static/heatmaps/{relative}"


def _encode_image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


@router.post("/", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    heatmap: bool = Query(
        default=False,
        description="Show Grad-CAM heatmap overlay on the original image.",
    ),
    include_base64: bool = Query(
        default=False,
        description="Include image content as base64 in response.",
    ),
    suspicious_only: bool = Query(
        default=True,
        description="Highlight only high-suspicion regions from Grad-CAM.",
    ),
    suspicious_threshold: float = Query(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Minimum Grad-CAM activation threshold for suspicious-only highlight.",
    ),
):
    # Validate file type and persist the upload to a temporary path.
    ext = _validate_extension(file.filename)
    tmp_path = _save_upload(file, ext)

    try:
        # Run inference first, then optionally generate Grad-CAM visualization.
        original_img, input_tensor = preprocess_image(str(tmp_path))
        label, conf = predict_image(
            str(tmp_path),
            str(MODEL_PATH),
            use_two_stage=get_predict_use_two_stage(),
            normal_threshold=get_predict_normal_threshold(),
        )
        label_meta = LABEL_INFO.get(label, _default_label_meta(label))
        label_description_en = label_meta.get("en", f"Model predicted class: {label}")
        label_description_th = label_meta.get("th", label_description_en)
        label_description = label_description_en

        original_out_path = tmp_path.with_name(f"{tmp_path.stem}_original{ext}")
        cv2.imwrite(str(original_out_path), cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))

        heatmap_out_path = None
        display_path = original_out_path

        if heatmap:
            # suspicious_only=True highlights only high-activation regions.
            overlay = build_gradcam_overlay(
                model_path=str(MODEL_PATH),
                input_tensor=input_tensor,
                original_img=original_img,
                focus_suspicious_only=suspicious_only,
                alpha=0.4,
                suspicious_threshold=suspicious_threshold,
            )
            heatmap_out_path = tmp_path.with_name(f"{tmp_path.stem}_gradcam{ext}")
            cv2.imwrite(
                str(heatmap_out_path),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )
            display_path = heatmap_out_path

    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return PredictResponse(
        predicted_label=label,
        label_code=label_meta["code"],
        label_description=label_description,
        label_description_en=label_description_en,
        label_description_th=label_description_th,
        confidence=conf,
        heatmap_enabled=heatmap,
        include_base64=include_base64,
        original_image_path=str(original_out_path),
        original_image_url=_to_public_heatmap_url(original_out_path),
        display_image_path=str(display_path),
        display_image_url=_to_public_heatmap_url(display_path),
        heatmap_path=str(heatmap_out_path) if heatmap_out_path else None,
        heatmap_url=_to_public_heatmap_url(heatmap_out_path) if heatmap_out_path else None,
        original_image_base64=_encode_image_base64(original_out_path) if include_base64 else None,
        display_image_base64=_encode_image_base64(display_path) if include_base64 else None,
        heatmap_image_base64=_encode_image_base64(heatmap_out_path) if include_base64 and heatmap_out_path else None,
    )
