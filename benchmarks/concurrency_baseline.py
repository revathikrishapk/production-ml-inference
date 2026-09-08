import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tritonclient.http as httpclient


TRITON_URL = "localhost:8000"
MODEL_NAME = "classifier"
MODEL_VERSION = "1"

NUM_REQUESTS = 100
CONCURRENCY = 8


def send_request():
    client = httpclient.InferenceServerClient(
        url=TRITON_URL
    )

    input_data = np.random.randn(
        1, 4
    ).astype(np.float32)

    input_tensor = httpclient.InferInput(
        "input",
        input_data.shape,
        "FP32"
    )

    input_tensor.set_data_from_numpy(
        input_data
    )

    output_tensor = httpclient.InferRequestedOutput(
        "output"
    )

    start = time.perf_counter()

    client.infer(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        inputs=[input_tensor],
        outputs=[output_tensor]
    )

    end = time.perf_counter()

    return (end - start) * 1000


start = time.perf_counter()

with ThreadPoolExecutor(
    max_workers=CONCURRENCY
) as executor:

    latencies = list(
        executor.map(
            lambda _: send_request(),
            range(NUM_REQUESTS)
        )
    )

total_time = time.perf_counter() - start

latencies = np.array(latencies)


print("Requests:", NUM_REQUESTS)
print("Concurrency:", CONCURRENCY)

print(
    "Mean:",
    np.mean(latencies),
    "ms"
)

print(
    "p50:",
    np.percentile(latencies, 50),
    "ms"
)

print(
    "p95:",
    np.percentile(latencies, 95),
    "ms"
)

print(
    "p99:",
    np.percentile(latencies, 99),
    "ms"
)

print(
    "Max:",
    np.max(latencies),
    "ms"
)

print(
    "Throughput:",
    NUM_REQUESTS / total_time,
    "req/s"
)