# src/app/routes/train.py
from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.modules.train_module_res50 import train_model
from pydantic import BaseModel, Field
import os
import json

router = APIRouter()

class TrainRequest(BaseModel):
    batch_size: int = Field(..., gt=0, alias="batchSize", description="Batch size for training")
    epochs: int = Field(..., gt=0, description="Number of epochs for training")
    learning_rate: float = Field(..., gt=0, alias="learningRate", description="Learning rate for the optimizer")


@router.post("/start")
async def start_training(
    params: TrainRequest, 
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(train_model, params.epochs, params.batch_size, params.learning_rate)
    return {"message": "Training started"}

@router.post("/stop")
async def stop_training():
    
    return {"message": "Training stop request sent"}

@router.get("/metrics")
async def get_training_metrics():
    # กำหนด path ของไฟล์ training_history.json
    # สมมุติว่าเราได้บันทึกไว้ที่ project_root/logs/training_history.json
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    history_path = os.path.join(project_root, "logs", "training_history.json")
    
    if not os.path.exists(history_path):
        raise HTTPException(status_code=404, detail="Training metrics not found")
    
    try:
        with open(history_path, "r") as f:
            metrics = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metrics: {e}")
    
    return metrics
