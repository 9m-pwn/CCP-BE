# src/app/routes/predict.py
import os
import shutil
import tempfile
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from ..predict_module import predict_image as predict_seq
from app.utils.preprocessing_util import make_input_tensor, load_image, preprocess_for_resnet50
from app.utils.prediction_util    import load_trained_model, predict_label_and_confidence
from app.utils.heatmap_util       import make_gradcam_heatmap, overlay_heatmap_on_image

# Create router instance
router = APIRouter(prefix="/predict", tags=["predict"])

# configurable
ROOT            = Path(__file__).parent.parent
LOG_ROOT       = Path(__file__).resolve().parents[2]
MODEL_RES50     = ROOT / "model_from_resnet50_cbam.keras"
LAST_CONV_LAYER = "conv5_block3_out"
VALID_EXTS      = {".jpg", ".jpeg", ".png"}


class PredictionResponse(BaseModel):
    predicted_label: str
    confidence: float


class HeatmapResponse(PredictionResponse):
    heatmap_path: str


def _validate_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in VALID_EXTS:
        raise HTTPException(400, f"Invalid file extension: {ext}")
    return ext


def _save_upload(file: UploadFile, ext: str) -> Path:

    upload_dir = os.path.join(LOG_ROOT, "logs", "heatmaps", "predict")
    os.makedirs(upload_dir, exist_ok=True)  

    # Use NamedTemporaryFile for safety and easy cleanup
    tmp = tempfile.NamedTemporaryFile( 
        prefix="upload_",
        suffix=ext,
        delete=False,
        dir=upload_dir,
    )
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.flush()
    except Exception as e:
        tmp.close()
        os.unlink(tmp.name)
        raise HTTPException(500, f"Cannot save upload: {e}")
    finally:
        file.file.close()
    return Path(tmp.name)


@router.post("/", response_model=PredictionResponse)
async def predict_sequential(file: UploadFile = File(...)):
    ext = _validate_extension(file.filename)
    tmp_path = _save_upload(file, ext)

    try:
        label, conf = predict_seq(str(tmp_path), str(MODEL_RES50))
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
        # 1) Load model
        model = load_trained_model(str(MODEL_RES50))

        # 2) Prepare input tensor
        inp = make_input_tensor(str(tmp_path), preprocess_for_resnet50)

        # 3) Predict label + confidence
        idx, conf = predict_label_and_confidence(model, inp)
        from app.utils.prediction_util import CLASS_MAP
        label = CLASS_MAP.get(idx, str(idx))

        # 4) Generate heatmap + overlay
        orig    = load_image(str(tmp_path))  # numpy array RGB 0–255
        heat    = make_gradcam_heatmap(model, inp, LAST_CONV_LAYER)
        overlay = overlay_heatmap_on_image(orig, heat)

        # 5) Save heatmap overlay
        out_file = tmp_path.with_name(f"{tmp_path.stem}_gradcam{ext}")
        import cv2
        cv2.imwrite(str(out_file), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Heatmap prediction error: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return HeatmapResponse(
        predicted_label=label,
        confidence=round(conf * 100, 2),
        heatmap_path=str(out_file)
    )