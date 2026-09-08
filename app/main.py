import os
import time

from prometheus_client import Counter, Histogram, make_asgi_app
from fastapi import Request
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tritonclient.http as httpclient


app = FastAPI(
    title="Production ML Inference API",
    version="1.0.0"
)

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request latency in seconds",
    ["method", "endpoint"]
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
MODEL_NAME = "classifier"
MODEL_VERSION = "1"


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "healthy"}

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):

    start_time = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start_time

    if not request.url.path.startswith("/metrics"):
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)

    return response

@app.get("/ready")
def ready():
    try:
        client = httpclient.InferenceServerClient(url=TRITON_URL)

        if not client.is_server_ready():
            raise HTTPException(
                status_code=503,
                detail="Triton server is not ready"
            )

        if not client.is_model_ready(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION
        ):
            raise HTTPException(
                status_code=503,
                detail="Model is not ready"
            )

        return {
            "status": "ready",
            "model": MODEL_NAME,
            "version": MODEL_VERSION
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Triton server unavailable"
        )


@app.post("/predict")
def predict(request: PredictionRequest):

    if len(request.features) != 4:
        raise HTTPException(
            status_code=400,
            detail="Expected exactly 4 features"
        )

    try:
        client = httpclient.InferenceServerClient(
            url=TRITON_URL
        )

        input_data = np.array(
            [request.features],
            dtype=np.float32
        )

        input_tensor = httpclient.InferInput(
            "input",
            input_data.shape,
            "FP32"
        )

        input_tensor.set_data_from_numpy(input_data)

        output_tensor = httpclient.InferRequestedOutput(
            "output"
        )

        response = client.infer(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            inputs=[input_tensor],
            outputs=[output_tensor]
        )

        prediction = response.as_numpy("output")

        return {
            "model": MODEL_NAME,
            "version": MODEL_VERSION,
            "prediction": prediction.tolist()
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Inference service unavailable: {str(e)}"
        )