# Cervical Cancer Detection API

FastAPI service for cervical cell image prediction and model training.

## Quickstart

1. Create or update the Conda environment.

```bash
conda env create -f environment.yml
conda activate poc-env
conda env update -f environment.yml
```

2. Start Redis (Docker).

```bash
docker compose up -d redis
docker compose ps redis
```

Or use helper scripts:

```powershell
.\scripts\redis_up.ps1
```

```bash
./scripts/redis_up.sh
```

3. Start the API from project root.

```bash
python run_api.py
```

Or use helper scripts:

```powershell
.\scripts\run_api.ps1
```

```bash
./scripts/run_api.sh
```

4. Open API docs.

```text
http://localhost:8000/docs
```

## Runtime config

Optional environment variables:

- `API_HOST` (default: `0.0.0.0`)
- `API_PORT` (default: `8000`)
- `API_RELOAD` (default: `true`)
- `REDIS_URL` (default: `redis://localhost:6379`)

Example:

```powershell
$env:API_PORT="8080"
$env:REDIS_URL="redis://127.0.0.1:6379"
python run_api.py
```

## Docker Redis

Required only if you need Redis-backed training status features.

Start:

```bash
docker compose up -d redis
```

Stop:

```bash
docker compose stop redis
```

Logs:

```bash
docker compose logs -f redis
```

Health check from app side:

- Call `GET /health`
- Confirm `redis_connected` is `true`

## API endpoints

- `GET /health`
- `POST /predict/` (`heatmap=true|false`)
- `GET /static/heatmaps/{relative_path}`
- `POST /train/start`
- `POST /train/stop`
- `GET /train/metrics`
- `GET /train/metrics/plot`
- `WS /ws/train-status`

## Request samples

`POST /train/start`

```json
{
  "batchSize": 32,
  "epochs": 20,
  "learningRate": 0.0001
}
```

`POST /predict/` uses:

- multipart file upload field: `file`
- query boolean `heatmap` (default `false`)
- query boolean `include_base64` (default `false`)

Response fields:

- `predicted_label`
- `label_code`
- `label_description`
- `label_description_en`
- `label_description_th`
- `confidence`
- `heatmap_enabled`
- `include_base64`
- `original_image_path`
- `original_image_url`
- `display_image_path` (original image when `heatmap=false`, overlay image when `heatmap=true`)
- `display_image_url`
- `heatmap_path` (only when `heatmap=true`)
- `heatmap_url` (only when `heatmap=true`)
- `original_image_base64` (when `include_base64=true`)
- `display_image_base64` (when `include_base64=true`)
- `heatmap_image_base64` (when `include_base64=true` and `heatmap=true`)

## Project structure

```text
.
|- docker-compose.yml
|- environment.yml
|- run_api.py
|- scripts/
|  |- redis_up.ps1
|  |- redis_down.ps1
|  |- redis_up.sh
|  |- redis_down.sh
|- src/
|  |- app/
|  |  |- main.py
|  |  |- settings.py
|  |  |- routes/
|  |  |- modules/
|  |  |- utils/
```

## Notes

- API can start without Redis. In that case `GET /health` returns `redis_connected: false`.
- If Redis is unavailable, `WS /ws/train-status` still responds with status `redis_unavailable`.
- Expected training data location is under `src/data/train/`.
- `GET /train/metrics` reads `src/logs/history.json` (and falls back to `src/logs/training_history.json` for compatibility).
- `GET /train/metrics` also returns `plot_ready` with `epochs` and `series` for frontend charting.
- `GET /train/metrics/plot` returns a PNG chart for quick visualization.
- Training now writes `src/app/model_from_resnet50_cbam.labels.json` for class index to label mapping used by prediction.
- Training excludes label `Not done` before fitting. Retrain the model to apply this filter to predictions.
