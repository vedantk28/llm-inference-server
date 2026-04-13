# 🚀 LLM Inference Server

A production-grade LLM inference gateway built with **FastAPI** that optimizes local GPU usage via Ollama/vLLM, automatically falls back to Azure OpenAI when the GPU is saturated, enforces token budgets, and exposes real-time observability through Prometheus + Grafana.

**Built for:** NVIDIA RTX 3050 (4GB VRAM) — but works on any NVIDIA GPU.

## 📊 Benchmark Results (RTX 3050 Laptop GPU)

| Metric | Mistral 7B (CPU/GPU split) | Qwen 1.5B (100% GPU) |
|---|---|---|
| **Avg Latency** | 19,380 ms | **1,067 ms** |
| **P50 Latency** | 14,810 ms | **616 ms** |
| **Throughput** | 8 tok/s | **131 tok/s** |
| **GPU Utilization** | 65% GPU / 35% CPU | **100% GPU** |
| **VRAM Usage** | 3,083 MB / 4,096 MB | **2,000 MB / 4,096 MB** |

> **Key insight:** A smaller model (1.5B) that fits entirely in VRAM is **16× faster** than a larger model (7B) that spills to CPU.

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────────┐
                    │            FastAPI Gateway :8000         │
                    │                                          │
  Client ──────────▶│  Token Budget ──▶ Smart Router           │
  (curl/app)        │  Check (400)      │                      │
                    │                   ├──▶ Local Backend      │──▶ Ollama :11434
                    │                   │    (queue < 4 AND     │    or vLLM :8001
                    │                   │     GPU < 85%)        │
                    │                   │                       │
                    │                   └──▶ Azure OpenAI       │──▶ Azure API
                    │                       (fallback)          │
                    │                                          │
                    │  /metrics ──▶ Prometheus :9090 ──▶ Grafana :3000
                    └─────────────────────────────────────────┘
```

### Smart Routing — Dual-Signal Saturation Detection

| Signal | Cost | When it triggers |
|---|---|---|
| **Active request counter** (in-memory) | ~0 ns | Local queue ≥ `MAX_CONCURRENT_LOCAL` |
| **GPU utilization** (pynvml, 5s cache) | ~1 ms | GPU util > `GPU_SATURATION_THRESHOLD` |

The counter catches inference overload instantly. The GPU monitor catches external load (e.g. a training job running alongside).

---

## 📁 Project Structure

```
llm-inference-server/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, lifespan, middleware
│   ├── config.py                # Pydantic Settings (env-based config)
│   ├── routers/
│   │   ├── chat.py              # POST /v1/chat/completions (OpenAI-compatible)
│   │   └── benchmark.py         # POST /benchmark, GET /benchmark/results
│   ├── backends/
│   │   ├── base.py              # Abstract backend interface
│   │   ├── ollama_backend.py    # Ollama HTTP client + streaming
│   │   ├── vllm_backend.py      # vLLM OpenAI-compat client
│   │   └── azure_backend.py     # Azure OpenAI SDK client
│   ├── middleware/
│   │   └── token_budget.py      # Token counting + budget enforcement
│   ├── routing/
│   │   └── smart_router.py      # Request queue + GPU-aware routing
│   ├── gpu/
│   │   └── monitor.py           # pynvml GPU monitor with 5s TTL cache
│   ├── metrics/
│   │   └── prometheus.py        # Custom Prometheus counters/histograms
│   └── utils/
│       └── tokenizer.py         # Hybrid tokenizer (AutoTokenizer + tiktoken)
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       └── provisioning/        # Auto-provisioned datasource + dashboard
├── docker-compose.yml           # Prometheus + Grafana stack
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- NVIDIA GPU with drivers loaded (`nvidia-smi` should work)
- [Ollama](https://ollama.com/) installed and running
- Docker (optional, for Prometheus + Grafana)

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/llm-inference-server.git
cd llm-inference-server

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env if needed (defaults work for Ollama)

# Pull a model that fits in your VRAM
ollama pull qwen2.5:1.5b
```

### Run

```bash
# Start the gateway
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Test

```bash
# Health check
curl http://localhost:8000/health | python3 -m json.tool

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Streaming (tokens appear as generated)
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Count from 1 to 10"}],
    "max_tokens": 200,
    "stream": true
  }'

# Benchmark
curl -X POST http://localhost:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain recursion", "num_requests": 5, "concurrency": 2}'

# Prometheus metrics
curl http://localhost:8000/metrics | grep llm_
```

### Start Monitoring Stack (Optional)

```bash
docker compose up -d

# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)
# Dashboard is auto-provisioned
```

> **Networking Note:** The monitoring stack uses `network_mode: "host"` (Linux native). This provides production-level, zero-latency metric scraping without Docker NAT overlays.

---

## 📡 API Reference

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat (streaming + non-streaming) |
| `POST` | `/benchmark` | Run structured load test |
| `GET` | `/benchmark/results` | Last benchmark results |
| `GET` | `/health` | System health, GPU status, routing info |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Interactive Swagger UI |

### Response Headers

Every response includes:

| Header | Example | Description |
|---|---|---|
| `X-Tokens-Used` | `42` | Total tokens consumed |
| `X-Tokens-Remaining` | `4054` | Budget remaining |
| `X-Tokens-Budget` | `4096` | Configured budget |
| `X-Backend` | `ollama` | Backend that served the request |
| `X-Queue-Depth` | `1` | Current active local requests |
| `X-Duration-Ms` | `616` | Request latency |

### Token Budget Enforcement

Requests exceeding `MAX_TOKENS_BUDGET` are rejected with **HTTP 400**:

```json
{
  "detail": {
    "error": "Token budget exceeded",
    "message": "Token budget exceeded: 5012 tokens requested (12 input + 5000 max output) exceeds budget of 4096",
    "requested_input_tokens": 12,
    "requested_max_tokens": 5000,
    "total_requested": 5012,
    "budget": 4096
  }
}
```

> **Why 400, not 429?** HTTP 429 means "Too Many Requests" (rate limiting). Token budget violations are request validation errors — the request itself is invalid regardless of how slowly you send it.

---

## ⚙️ Configuration

All settings via environment variables or `.env` file:

| Variable | Default | Description |
|---|---|---|
| `ACTIVE_BACKEND` | `ollama` | Backend: `ollama` or `vllm` |
| `DEFAULT_MODEL` | `qwen2.5:1.5b` | Default model for inference |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `VLLM_BASE_URL` | `http://localhost:8001` | vLLM server URL |
| `MAX_TOKENS_BUDGET` | `4096` | Max tokens (prompt + completion) per request |
| `GPU_SATURATION_THRESHOLD` | `0.85` | GPU util threshold for Azure fallback |
| `MAX_CONCURRENT_LOCAL` | `4` | Max concurrent local requests before fallback |
| `GPU_UTIL_CACHE_TTL` | `5.0` | Seconds to cache GPU utilization readings |
| `HF_TOKENIZER_MODEL` | — | HuggingFace model ID for accurate token counting |
| `AZURE_OPENAI_ENDPOINT` | — | Azure OpenAI endpoint (optional) |
| `AZURE_OPENAI_API_KEY` | — | Azure OpenAI API key (optional) |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | — | Azure deployment name (optional) |

---

## 📈 Prometheus Metrics

| Metric | Type | Labels |
|---|---|---|
| `llm_request_total` | Counter | `backend`, `model`, `status` |
| `llm_request_duration_seconds` | Histogram | `backend`, `model` |
| `llm_tokens_input_total` | Counter | `backend`, `model` |
| `llm_tokens_output_total` | Counter | `backend`, `model` |
| `llm_cost_dollars_total` | Counter | `backend`, `model` |
| `llm_gpu_utilization_percent` | Gauge | — |
| `llm_gpu_memory_used_mb` | Gauge | — |
| `llm_active_requests` | Gauge | `backend` |
| `llm_request_rejected_total` | Counter | `reason` |

---

## 🔀 vLLM Backend (Optional)

For production throughput with **PagedAttention** and **continuous batching**:

```bash
# Launch vLLM with aggressive memory tuning for 4GB VRAM
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct-AWQ \
    --quantization awq \
    --dtype float16 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 2048 \
    --max-num-seqs 4 \
    --kv-cache-dtype fp8 \
    --port 8001

# Then switch in .env:
# ACTIVE_BACKEND=vllm
```

---

## 🧠 Key Design Decisions

| Decision | Rationale |
|---|---|
| **pynvml** over `nvidia-smi` subprocess | ~1ms vs ~200ms per GPU query — critical under load |
| **5s TTL cache** on GPU readings | Avoids hammering pynvml hundreds of times/sec |
| **On-Demand GPU Metric Polling** | Prometheus GPU variables update natively when `/health` is polled, minimizing background tracking overhead |
| **In-memory request counter** as primary routing signal | Zero-cost compared to GPU polling for every request |
| **`AutoTokenizer`** for local models | tiktoken `cl100k_base` is ~10-20% off for Qwen/Mistral tokenizers |
| **HTTP 400** for token budget (not 429) | 429 is rate limiting; budget violations are request validation errors |
| **SSE streaming** | Without it, users wait 10+ seconds of silence for full generation |

---

## 🎓 Learning Objectives

This project covers:

- **GGUF INT4/INT8 quantization** — model weights in 4-bit, reducing size 4×
- **PagedAttention** (vLLM) — virtual memory-style KV cache management
- **Continuous batching** — serving multiple users concurrently on one GPU
- **KV cache quantization** — fp8 KV cache saves ~50% memory
- **pynvml GPU monitoring** — direct C bindings to NVIDIA Management Library
- **Token budgeting** — server-side enforcement of token limits
- **Hybrid tokenization** — AutoTokenizer for local models, tiktoken for OpenAI
- **SSE streaming** — real-time token delivery via Server-Sent Events
- **Prometheus + Grafana observability** — production monitoring pipeline

---

## 📝 License

MIT
