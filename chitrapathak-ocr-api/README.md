# Chitrapathak-2 OCR API

Production-ready **FastAPI OCR microservice** using the **Chitrapathak-2 Vision-Language Model** for transcribing historical Sanskrit/Devanagari manuscripts.

- **Single OCR pass** on original color images — no preprocessing, no grayscale, no CLAHE
- **Qwen2.5-VL architecture** with singleton model loading
- **GPU & CPU** support with auto-detection
- **Docker** ready for **AWS EC2** deployment
- Detailed **performance metrics** per request

---

## Quick Start

### 1. Clone & Configure

```bash
cd chitrapathak-ocr-api
cp .env.example .env
# Edit .env if needed (e.g., set HF_TOKEN for gated models)
```

### 2. Run with Docker (Recommended)

**Build the image:**

```bash
docker build -t chitrapathak-ocr .
```

**Run (CPU):**

```bash
docker run -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  chitrapathak-ocr
```

**Run (GPU — requires NVIDIA Container Toolkit):**

```bash
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  chitrapathak-ocr
```

### 3. Run with Docker Compose

```bash
# CPU
docker compose --profile cpu up -d

# GPU
docker compose --profile gpu up -d
```

### 4. Run Locally (without Docker)

```bash
pip install -r requirements.txt
chmod +x start.sh
./start.sh
```

---

## API Endpoints

### POST /ocr

Upload an image and get the transcribed Devanagari text.

```bash
curl -X POST http://localhost:8000/ocr \
  -F "file=@manuscript.jpg" \
  -F "max_new_tokens=2048"
```

**Response:**

```json
{
  "request_id": "b7d12a9f",
  "status": "success",
  "text": "देवनागरी OCR परिणाम...",
  "metrics": {
    "device": "cuda",
    "queue_wait": 0.15,
    "image_load": 0.02,
    "tokenization": 0.04,
    "inference": 8.73,
    "decode": 0.02,
    "total": 8.96,
    "input_tokens": 522,
    "output_tokens": 314,
    "tokens_per_second": 35.97,
    "cpu": 27.0,
    "ram": 41.0,
    "gpu": 96.0,
    "gpu_memory": 78.5
  }
}
```

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "queue_size": 0,
  "uptime": 3600.5,
  "model_name": "krutrim-ai-labs/Chitrapathak-2"
}
```

### GET /metrics

```bash
curl http://localhost:8000/metrics
```

```json
{
  "total_requests": 150,
  "success": 148,
  "failed": 2,
  "avg_inference_time": 9.12,
  "avg_total_time": 9.45,
  "cpu": 35.0,
  "ram": 52.0,
  "gpu": 45.0,
  "gpu_memory": 68.2,
  "queue_size": 1,
  "device": "cuda"
}
```

---

## AWS EC2 Deployment

### Recommended Instance

| Type | vCPUs | RAM | GPU | Use Case |
|------|-------|-----|-----|----------|
| `g4dn.xlarge` | 4 | 16 GB | T4 (16 GB) | Production (GPU) |
| `g5.xlarge` | 4 | 16 GB | A10G (24 GB) | High-perf Production |
| `t3.2xlarge` | 8 | 32 GB | None | CPU-only (slow) |

### Setup on EC2

```bash
# 1. SSH into your EC2 instance
ssh -i your-key.pem ubuntu@<ec2-public-ip>

# 2. Install Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo systemctl start docker
sudo usermod -aG docker $USER
newgrp docker

# 3. (GPU only) Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 4. Clone or copy project
git clone <your-repo-url> chitrapathak-ocr-api
cd chitrapathak-ocr-api

# 5. Configure
cp .env.example .env
# Edit .env as needed

# 6. Build & Run
docker build -t chitrapathak-ocr .

# CPU
docker run -d -p 8000:8000 --name ocr-api chitrapathak-ocr

# GPU
docker run -d --gpus all -p 8000:8000 --name ocr-api chitrapathak-ocr

# 7. Verify
curl http://localhost:8000/health

# 8. Test OCR
curl -X POST http://localhost:8000/ocr \
  -F "file=@test_manuscript.jpg"
```

### Security Group Rules

| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 22 | TCP | Your IP | SSH |
| 8000 | TCP | 0.0.0.0/0 | API |

---

## Project Structure

```
chitrapathak-ocr-api/
├── app/
│   ├── __init__.py        # Package init
│   ├── main.py            # FastAPI app, endpoints, lifespan
│   ├── model.py           # Singleton model loader
│   ├── inference.py       # Single-pass OCR pipeline
│   ├── config.py          # Pydantic settings (.env)
│   ├── logger.py          # JSON logging + daily rotation
│   ├── metrics.py         # System metrics (CPU/RAM/GPU)
│   ├── queue_manager.py   # ThreadPoolExecutor + GPU lock
│   ├── schemas.py         # Pydantic request/response models
│   └── utils.py           # Helpers (request ID, validation)
├── uploads/               # Uploaded images
├── outputs/               # OCR result text files
├── logs/                  # Daily rotating log files
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── start.sh
└── README.md
```

---

## Configuration

All settings are configurable via `.env` file or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `krutrim-ai-labs/Chitrapathak-2` | HuggingFace model ID |
| `MAX_NEW_TOKENS` | `2048` | Max tokens per inference |
| `MAX_WORKERS` | `2` | Concurrent worker threads |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server bind port |
| `HF_TOKEN` | *(none)* | HuggingFace auth token |

---

## Tech Stack

- **Python 3.11** + **FastAPI** + **Uvicorn**
- **PyTorch** + **Transformers** + **qwen-vl-utils**
- **Chitrapathak-2** (Qwen2.5-VL architecture)
- **Pillow** for image handling
- **psutil** + **pynvml** for system metrics
- **Docker** + **Docker Compose**
- **Pydantic** for validation

---

## License

This project uses the Chitrapathak-2 model which is distributed under the [Krutrim Community License Agreement v1.0](https://huggingface.co/krutrim-ai-labs/Chitrapathak-2).
