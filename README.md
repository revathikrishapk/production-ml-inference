# Production ML Inference Platform

A production-style machine learning inference platform for serving an
ONNX model through **NVIDIA Triton Inference Server**, exposed through a
**FastAPI** gateway, containerized with **Docker Compose**, and
monitored with **Prometheus**.

The project focuses on the engineering problems that appear after a
model has been trained: model serving, API design, batching, health
checks, failure handling, observability, reproducible benchmarking, and
containerized deployment.

------------------------------------------------------------------------

## Architecture

``` text
                         ┌──────────────────────┐
                         │        Client        │
                         │  HTTP / JSON Request │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       FastAPI        │
                         │      :8080           │
                         │                      │
                         │  /predict            │
                         │  /health             │
                         │  /ready              │
                         │  /metrics/            │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Triton Server      │
                         │      :8000           │
                         │                      │
                         │  Model Scheduler     │
                         │  Dynamic Batching     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   ONNX Runtime       │
                         │      CPU             │
                         │                      │
                         │    classifier        │
                         └──────────────────────┘

             ┌──────────────────────┐
             │     Prometheus      │
             │       :9090         │
             │                      │
             │ FastAPI metrics      │
             │ Triton metrics       │
             └──────────────────────┘
```

### Request flow

``` text
Client
  │
  │ POST /predict
  ▼
FastAPI
  │
  │ Triton HTTP inference request
  ▼
Triton Server
  │
  │ Scheduler + dynamic batching
  ▼
ONNX Runtime
  │
  ▼
Prediction
  │
  ▼
FastAPI
  │
  ▼
Client
```

------------------------------------------------------------------------

## Why this project?

A machine learning model is not production-ready simply because it
produces accurate predictions.

A production inference system also needs to address:

-   How is the model served?
-   How does the application communicate with the model server?
-   How are requests validated?
-   What happens when the model server is unavailable?
-   How do we monitor request latency and traffic?
-   Can multiple requests be batched efficiently?
-   How is the system packaged for deployment?
-   Can performance improvements be measured objectively?

This project implements those concerns in a small, reproducible system.

------------------------------------------------------------------------

## Key Features

### Model serving

-   PyTorch-trained model exported to ONNX
-   NVIDIA Triton Inference Server
-   ONNX Runtime backend
-   CPU inference
-   Explicit model repository structure
-   Triton model readiness checks

### API layer

-   FastAPI inference gateway
-   JSON request validation using Pydantic
-   `/predict` endpoint
-   `/health` endpoint
-   `/ready` endpoint
-   `/metrics/` endpoint
-   Proper HTTP error responses

### Performance

-   Triton dynamic batching
-   Concurrent inference benchmarking
-   p50, p95 and p99 latency measurement
-   Throughput measurement
-   A/B comparison of dynamic batching
-   End-to-end FastAPI → Triton benchmarking

### Reliability

-   Input validation
-   Model readiness checks
-   Dependency failure handling
-   HTTP 503 responses when Triton is unavailable
-   Recovery verification after Triton restarts

### Observability

-   Prometheus metrics
-   FastAPI request counter
-   FastAPI request latency histogram
-   Triton inference metrics
-   Scraping configuration for both services

### Deployment

-   Dockerfile for FastAPI
-   Docker Compose
-   Triton container
-   Prometheus container
-   Container-to-container service discovery

------------------------------------------------------------------------

## Technology Stack

  Component          Technology
  ------------------ --------------------------------
  Model              ONNX
  Model runtime      ONNX Runtime
  Model server       NVIDIA Triton Inference Server
  API                FastAPI
  Validation         Pydantic
  Language           Python
  Containerization   Docker
  Orchestration      Docker Compose
  Monitoring         Prometheus
  Benchmarking       Python + ThreadPoolExecutor
  Communication      HTTP

------------------------------------------------------------------------

## Project Structure

``` text
production-ml-inference/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── model_repository/
│   └── classifier/
│       ├── config.pbtxt
│       └── 1/
│           └── model.onnx
│
├── client/
│   └── inference_client.py
│
├── benchmarks/
│   ├── baseline.py
│   ├── concurrency_baseline.py
│   └── final_benchmark.py
│
├── prometheus/
│   └── prometheus.yml
│
├── results/
│   ├── comparison.csv
│   └── final_benchmark.csv
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

------------------------------------------------------------------------

# Getting Started

## Prerequisites

Install:

-   Python 3.10+
-   Docker Desktop
-   Git

Verify the installations:

``` powershell
python --version
docker --version
docker compose version
git --version
```

No NVIDIA GPU is required for the current CPU deployment.

------------------------------------------------------------------------

# Run with Docker Compose

The recommended way to run the complete platform is Docker Compose.

## 1. Clone the repository

``` powershell
git clone https://github.com/revathikrishapk/production-ml-inference.git
cd production-ml-inference
```

## 2. Start the platform

``` powershell
docker compose up -d
```

Check the containers:

``` powershell
docker compose ps
```

Expected services:

``` text
ml-api
triton-server
prometheus
```

Triton should eventually report a healthy state.

------------------------------------------------------------------------

# Service Endpoints

  Service          URL                               Purpose
  ---------------- --------------------------------- --------------------
  FastAPI          `http://localhost:8080`           Inference API
  Triton HTTP      `http://localhost:8000`           Model serving
  Triton gRPC      `localhost:8001`                  gRPC model serving
  Triton metrics   `http://localhost:8002/metrics`   Triton monitoring
  Prometheus       `http://localhost:9090`           Metrics UI

------------------------------------------------------------------------

# API

## Health check

``` powershell
curl.exe -i http://localhost:8080/health
```

Expected response:

``` json
{
  "status": "healthy"
}
```

The health endpoint confirms that the API process is running.

------------------------------------------------------------------------

## Readiness check

``` powershell
curl.exe -i http://localhost:8080/ready
```

Expected response:

``` json
{
  "status": "ready",
  "model": "classifier",
  "version": "1"
}
```

The readiness endpoint verifies both:

1.  Triton is reachable.
2.  The `classifier` model is ready.

If Triton is unavailable, the API returns:

``` text
503 Service Unavailable
```

instead of incorrectly reporting that the service is ready.

------------------------------------------------------------------------

# Prediction

Send exactly four floating-point features:

``` powershell
curl.exe -X POST http://localhost:8080/predict `
  -H "Content-Type: application/json" `
  -d '{\"features\":[0.1,0.2,0.3,0.4]}'
```

Example response:

``` json
{
  "model": "classifier",
  "version": "1",
  "prediction": [
    [
      0.42140787839889526,
      -0.26393723487854004,
      -0.023611754179000854
    ]
  ]
}
```

------------------------------------------------------------------------

# Input Validation

The API expects exactly four features.

Invalid request:

``` powershell
curl.exe -i -X POST http://localhost:8080/predict `
  -H "Content-Type: application/json" `
  -d '{\"features\":[0.1,0.2]}'
```

Response:

``` text
HTTP/1.1 400 Bad Request
```

``` json
{
  "detail": "Expected exactly 4 features"
}
```

This prevents malformed input from reaching the inference server.

------------------------------------------------------------------------

# Triton Model Repository

The model follows the Triton model repository structure:

``` text
model_repository/
└── classifier/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

The model configuration specifies:

``` text
name: classifier
platform: onnxruntime_onnx
max_batch_size: 8
```

Dynamic batching is enabled using:

``` text
dynamic_batching {}
```

This allows Triton to combine compatible requests into batches before
model execution.

------------------------------------------------------------------------

# Dynamic Batching

One of the main experiments in this project was measuring the effect of
Triton's dynamic batching.

Two configurations were compared under the same workload:

-   100 requests
-   concurrency = 8
-   CPU inference
-   same model
-   same client implementation

## Results

  ----------------------------------------------------------------------------------------
  Configuration     Throughput   Mean (ms)    P50 (ms)    P95 (ms)    P99 (ms)    Max (ms)
                       (req/s)                                                 
  --------------- ------------ ----------- ----------- ----------- ----------- -----------
  Dynamic               349.00       18.69       16.65       33.86       39.11       50.50
  batching OFF                                                                 

  Dynamic           **416.04**   **15.63**   **14.32**   **28.82**   **36.88**   **38.12**
  batching ON                                                                  
  ----------------------------------------------------------------------------------------

### Improvement

Dynamic batching produced:

-   **19.2% higher throughput**
-   **14.9% lower p95 latency**
-   **24.5% lower maximum observed latency**

The improvement was measured rather than assumed.

------------------------------------------------------------------------

# Triton Batch Statistics

Triton exposes inference and execution counters.

For the dynamic batching experiment, the server reported approximately:

``` text
inference_count = 100
execution_count = 82
```

The approximate average batch size can be estimated as:

``` text
average batch size
≈ inference_count / execution_count

≈ 100 / 82

≈ 1.22
```

The benchmark also observed batches larger than one request,
demonstrating that Triton's scheduler was combining concurrent requests.

------------------------------------------------------------------------

# End-to-End Benchmark

The final benchmark measures the inference-serving path using the Triton
HTTP client:

``` text
Benchmark Client
      ↓
Triton HTTP
      ↓
ONNX Runtime
      ↓
Model
```

The final measured workload was:

``` text
Requests:     100
Concurrency:  8
```

## Final benchmark

  Metric                         Result
  ----------------- -------------------
  Requests                          100
  Concurrency                         8
  Mean latency                21.083 ms
  P50 latency                 15.692 ms
  P95 latency                 79.143 ms
  P99 latency                 86.609 ms
  Maximum latency             92.496 ms
  Throughput          **122.624 req/s**

The benchmark results are stored in:

``` text
results/final_benchmark.csv
```

The A/B comparison is stored in:

``` text
results/comparison.csv
```

------------------------------------------------------------------------

# Running the Benchmarks

## Baseline benchmark

``` powershell
python benchmarks/baseline.py
```

## Concurrency benchmark

``` powershell
python benchmarks/concurrency_baseline.py
```

## Final benchmark

``` powershell
python benchmarks/final_benchmark.py
```

The final benchmark writes:

``` text
results/final_benchmark.csv
```

------------------------------------------------------------------------

# Prometheus Monitoring

Prometheus collects metrics from both:

``` text
FastAPI
Triton
```

Prometheus configuration:

``` text
prometheus/prometheus.yml
```

Open:

``` text
http://localhost:9090
```

------------------------------------------------------------------------

## FastAPI Metrics

The API exposes Prometheus metrics at:

``` text
http://localhost:8080/metrics/
```

The application records:

``` text
api_requests_total
```

and:

``` text
api_request_duration_seconds
```

The request counter is labeled by:

``` text
method
endpoint
status
```

The latency histogram is labeled by:

``` text
method
endpoint
```

The `/metrics` endpoint itself is excluded from application request
metrics to avoid monitoring traffic polluting application measurements.

------------------------------------------------------------------------

# Triton Metrics

Triton exposes metrics at:

``` text
http://localhost:8002/metrics
```

Useful metrics include:

``` text
nv_inference_count
nv_inference_exec_count
```

These metrics can be queried directly:

``` powershell
curl.exe http://localhost:8002/metrics
```

Or through Prometheus.

Example PromQL:

``` promql
nv_inference_count
```

------------------------------------------------------------------------

# Reliability Testing

The system was explicitly tested under both valid and failure
conditions.

  Test                    Expected   Result
  ----------------------- ---------- --------
  Valid prediction        HTTP 200   Passed
  Invalid feature count   HTTP 400   Passed
  API health              HTTP 200   Passed
  Triton + model ready    HTTP 200   Passed
  Triton unavailable      HTTP 503   Passed
  Triton recovery         HTTP 200   Passed

### Dependency failure behavior

When Triton was stopped:

``` powershell
docker compose stop triton
```

the API readiness endpoint returned:

``` text
503 Service Unavailable
```

After Triton was restarted:

``` powershell
docker compose start triton
```

the system recovered and `/ready` returned HTTP 200.

This demonstrates dependency-aware readiness rather than a simple
process-level health check.

------------------------------------------------------------------------

# Docker Architecture

The application is separated into three containers:

``` text
┌─────────────────────────────────────────────┐
│              Docker Compose                 │
│                                             │
│  ┌────────────┐     ┌──────────────────┐   │
│  │   ml-api   │────▶│  triton-server   │   │
│  │   :8080    │     │     :8000        │   │
│  └────────────┘     └──────────────────┘   │
│          │                    │              │
│          │                    │              │
│          ▼                    ▼              │
│  ┌──────────────────────────────────────┐   │
│  │            Prometheus :9090          │   │
│  └──────────────────────────────────────┘   │
│                                             │
└─────────────────────────────────────────────┘
```

Within Docker Compose, FastAPI communicates with Triton using:

``` text
triton:8000
```

rather than:

``` text
localhost:8000
```

This is important because `localhost` inside the FastAPI container
refers to the FastAPI container itself.

------------------------------------------------------------------------

# Environment Configuration

The FastAPI application supports the Triton server address through:

``` text
TRITON_URL
```

The Docker Compose deployment uses:

``` text
TRITON_URL=triton:8000
```

The local development default is:

``` text
localhost:8000
```

This allows the same application code to work both locally and inside
Docker.

------------------------------------------------------------------------

# Local Development Without Docker

Create a virtual environment:

``` powershell
python -m venv .venv
```

Activate it:

``` powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

``` powershell
pip install -r requirements.txt
```

Start FastAPI:

``` powershell
fastapi dev app\main.py --port 8080
```

Triton still needs to be running separately.

For local development, the application defaults to:

``` text
localhost:8000
```

------------------------------------------------------------------------

# Stopping the Platform

Stop the containers:

``` powershell
docker compose down
```

Stop and remove containers while preserving local images:

``` powershell
docker compose down
```

To view logs:

``` powershell
docker compose logs
```

FastAPI logs:

``` powershell
docker compose logs api
```

Triton logs:

``` powershell
docker compose logs triton
```

Prometheus logs:

``` powershell
docker compose logs prometheus
```

------------------------------------------------------------------------

# Engineering Decisions

## Why Triton?

Triton provides a dedicated model-serving layer rather than coupling
model execution directly to the API.

This separates:

``` text
API responsibilities
```

from:

``` text
Model-serving responsibilities
```

The API handles:

-   request validation
-   HTTP responses
-   readiness
-   application metrics

Triton handles:

-   model loading
-   inference scheduling
-   batching
-   backend execution
-   model-serving metrics

------------------------------------------------------------------------

## Why ONNX?

ONNX provides a portable model representation and allows the model to be
served through ONNX Runtime.

This separates the training/export environment from the serving
environment.

------------------------------------------------------------------------

## Why FastAPI?

FastAPI provides:

-   typed request validation
-   automatic API documentation
-   asynchronous server support
-   straightforward HTTP endpoint development
-   easy integration with Prometheus metrics

------------------------------------------------------------------------

## Why Docker Compose?

Docker Compose makes the complete system reproducible:

``` text
FastAPI
+
Triton
+
Prometheus
```

can be started using:

``` powershell
docker compose up -d
```

This avoids manually configuring each service.

------------------------------------------------------------------------

## Why Prometheus?

Prometheus provides a standard metrics collection system for observing:

-   request volume
-   request latency
-   HTTP status codes
-   inference counts
-   model execution counts
-   serving behavior

------------------------------------------------------------------------

# Performance Optimization Methodology

The optimization process followed a measurement-driven approach:

``` text
Baseline
   ↓
Introduce dynamic batching
   ↓
Run identical workload
   ↓
Measure latency + throughput
   ↓
Compare results
   ↓
Keep the configuration that improves the workload
```

Rather than assuming that batching would improve performance, the
project measured the actual impact.

The benchmark showed that dynamic batching improved throughput while
also reducing the measured p95 latency for this workload.

------------------------------------------------------------------------

# Production Considerations

The current system intentionally stays small enough to run locally while
demonstrating production-oriented engineering patterns.

Potential next steps include:

-   CI/CD automation
-   automated benchmark execution
-   Kubernetes deployment
-   Grafana dashboards
-   authentication and authorization
-   rate limiting
-   model version management
-   cloud deployment
-   distributed inference
-   automated model rollout strategies

These are intentionally outside the scope of the current implementation.

------------------------------------------------------------------------

# Repository Contents

### `app/`

FastAPI application.

### `model_repository/`

Triton model repository containing the ONNX model and serving
configuration.

### `client/`

Python inference client for communicating with Triton.

### `benchmarks/`

Benchmark scripts used to measure serving performance.

### `prometheus/`

Prometheus scraping configuration.

### `results/`

Reproducible benchmark results stored as CSV files.

### `Dockerfile`

Container image definition for the FastAPI service.

### `docker-compose.yml`

Multi-container deployment configuration.

### `requirements.txt`

Python dependencies.

------------------------------------------------------------------------

# Example Workflow

A typical workflow looks like:

``` text
1. Train model
      ↓
2. Export model to ONNX
      ↓
3. Place model in Triton repository
      ↓
4. Configure Triton
      ↓
5. Start Docker Compose
      ↓
6. Verify health
      ↓
7. Verify readiness
      ↓
8. Send inference requests
      ↓
9. Collect Prometheus metrics
      ↓
10. Run benchmarks
      ↓
11. Compare serving configurations
```

------------------------------------------------------------------------

# Results Summary

  Area                                      Result
  ----------------------------------------- ------------------
  Model format                              ONNX
  Model server                              NVIDIA Triton
  Runtime                                   ONNX Runtime
  API                                       FastAPI
  Deployment                                Docker Compose
  Monitoring                                Prometheus
  Dynamic batching                          Enabled
  Benchmark concurrency                     8
  Dynamic batching throughput improvement   **19.2%**
  Dynamic batching p95 improvement          **14.9%**
  Final measured throughput                 **122.62 req/s**
  Final measured p95 latency                **79.14 ms**
  Failure handling                          HTTP 400 / 503
  Triton recovery tested                    Yes

------------------------------------------------------------------------

# What This Project Demonstrates

This project demonstrates practical experience with:

-   Machine learning model serving
-   ONNX model deployment
-   NVIDIA Triton Inference Server
-   ONNX Runtime
-   FastAPI
-   REST API design
-   Request validation
-   Dynamic batching
-   Concurrent inference
-   Latency and throughput benchmarking
-   Docker
-   Docker Compose
-   Prometheus monitoring
-   Health and readiness checks
-   Dependency failure handling
-   Production-oriented system design

------------------------------------------------------------------------

# Author

**Revathi Krishna**

MSc Artificial Intelligence & Machine Learning

GitHub: [revathikrishapk](https://github.com/revathikrishapk)

------------------------------------------------------------------------

# License

This project does not currently include a separate license file.

If this repository is intended for public reuse, add an appropriate
`LICENSE` file before presenting it as an open-source project.

------------------------------------------------------------------------

## Final Architecture Summary

``` text
                         PRODUCTION ML INFERENCE PLATFORM

                                  Client
                                    │
                                    ▼
                              ┌───────────┐
                              │  FastAPI  │
                              │   :8080   │
                              └─────┬─────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │   Triton Server    │
                         │      :8000         │
                         │                    │
                         │ Dynamic Batching   │
                         │ Model Scheduler    │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │   ONNX Runtime     │
                         │       CPU          │
                         └─────────┬──────────┘
                                   │
                                   ▼
                              Prediction

                 ┌────────────────────────────────┐
                 │          Prometheus             │
                 │              :9090              │
                 │                                │
                 │ FastAPI ──────┐               │
                 │ Triton ───────┴─ Metrics      │
                 └────────────────────────────────┘
```

**Built as a reproducible production-style ML inference system rather
than a standalone model demo.**
