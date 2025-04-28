# Cervical Cancer Detection API

ระบบฝึกและให้บริการ API สำหรับตรวจวินิจฉัยภาพเซลล์ปากมดลูก (Cervical Cancer Detection)  
โดยใช้โมเดล ResNet50 + CBAM และสามารถดูผล Grad‑CAM เพื่ออธิบายการตัดสินใจของโมเดล

---

## 📁 โครงสร้างโปรเจกต์

```
├── environment.yml          # ไฟล์คอนฟิก conda environment  
├── README.md                # ไฟล์นี้ 
├── src/ 
│   └── app/ 
│       ├── main.py          # FastAPI entrypoint 
│       ├── routes/ 
│       │   ├── predict.py   # เส้นทาง API สำหรับ predict & heatmap 
│       │   ├── train.py     # เส้นทาง API สำหรับสั่งเทรน 
│       │   └── train_ws.py  # WebSocket สำหรับสถานะการเทรน 
│       ├── modules/ 
│       │   ├── train_module_res50.py  # สคริปต์ฝึกโมเดล ResNet50+CBAM 
│       │   └── predict_module*.py     # ฟังก์ชันสำหรับรัน inference 
│       └── utils/          # เฮลเปอร์ต่าง ๆ (preprocessing, gradcam, redis) 
└── data/                   # (ไม่เก็บใน Git) โฟลเดอร์ข้อมูลดิบและ processed
```

---

## 🛠️ ติดตั้งและเตรียม environment

ใช้ Conda ตามไฟล์ `environment.yml` (ชื่อ env: `poc-env`)

```bash
# ในโฟลเดอร์โปรเจกต์ (ที่มี environment.yml)
conda env create -f environment.yml      # สร้าง env ใหม่
conda activate poc-env                   # หรือถ้ามี env อยู่แล้ว
conda env update -f environment.yml      # อัปเดตแพ็กเกจให้ตรงกับไฟล์
```

> **หมายเหตุ:**  
> ถ้าชื่อ env ซ้ำ ให้ลบก่อน (`conda env remove -n poc-env`) หรือสร้างชื่อใหม่ด้วย `-n`

---

## 🚀 รัน API (FastAPI + Uvicorn)


1. เข้าไปที่ src
```bash
cd src/app
```
2a. รันเป็นโมดูล Python (ใช้ตัวตรวจหา path ของ package อัตโนมัติ)
```bash
python -m app.main
```

2b. หรือถ้าอยากใช้ uvicorn ตรงๆ ก็
```bash
cd src/app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

จากนั้นเปิดเบราว์เซอร์ที่ [http://localhost:8000/docs](http://localhost:8000/docs) เพื่อดู Swagger UI และทดสอบ API

---

## 📦 API Endpoints

### 1. POST `/predict/`
รันโมเดล Sequential (รุ่นเก่า)

**Request:**  
- UploadFile (.jpg/.jpeg/.png)

**Response:**  
```json
{
  "predicted_label": "HSIL (CIN2/3)",
  "confidence": 87.52
}
```

### 2. POST `/predict/resnet50`
รันโมเดล ResNet50 (+ CBAM)

**Query params (ใหม่):**  
- `?heatmap=true|false`

**Response:**  
- ถ้า `heatmap=false` (default):  
  ```json
  {
    "predicted_label": "Normal",
    "confidence": 99.76
  }
  ```
- ถ้า `heatmap=true`:  
  ```json
  {
    "predicted_label": "HSIL (CIN2/3)",
    "confidence": 87.52,
    "heatmap_path": "/logs/heatmaps/predict/upload_xxx_gradcam.png"
  }
  ```

### 3. POST `/predict/heatmap`
สร้าง Grad‑CAM overlay พร้อม return path ไฟล์

---

## 🔄 ฝึกโมเดลบนเซิร์ฟเวอร์

### POST `/train/`
เริ่มการฝึกโมเดล (background task)

**Request body:**  
```json
{
  "epochs": 20,
  "batch_size": 32,
  "learning_rate": 0.0001
}
```

**Response:**  
- สถานะรับคำสั่งฝึก

### WebSocket `/train/ws`
Subscribe เพื่อรับสถานะการเทรนแบบ real‑time จาก Redis (publish ทุกครั้งหลังจบแต่ละ epoch)

```javascript
const ws = new WebSocket("ws://localhost:8000/train/ws");
ws.onmessage = e => {
  const status = JSON.parse(e.data);
  console.log("Train status:", status);
};
```

---

## 🔍 Grad‑CAM

- **ขณะฝึก:** callback จะสร้างไฟล์ heatmap ทุกจบ epoch ลงใน `logs/heatmaps/train`
- **ขณะพรีดิกต์:** ใส่ query param `heatmap=true` หรือเรียก endpoint `/predict/heatmap`

**โครงสร้างภาพ output:**  
- สีแดงเข้ม = บริเวณโมเดลให้ความสำคัญสูงสุด  
- สีฟ้าอ่อน = ความสำคัญต่ำ  

---

## 📂 จัดการไฟล์ใหญ่

- เก็บ `.keras`, `.h5`, raw images, logs ใน `.gitignore`
- ใช้ Git LFS ถ้าต้องการเก็บโมเดลหรือรูปขนาดใหญ่จริง ๆ