import time
import numpy as np
import tritonclient.http as httpclient


TRITON_URL = "localhost:8000"
MODEL_NAME = "classifier"
MODEL_VERSION = "1"

NUM_REQUESTS = 100


client = httpclient.InferenceServerClient(
    url=TRITON_URL
)


latencies_ms = []


for _ in range(NUM_REQUESTS):

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

    latency_ms = (end - start) * 1000

    latencies_ms.append(latency_ms)


latencies = np.array(latencies_ms)


print("Requests:", NUM_REQUESTS)

print(
    "Mean latency:",
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
    "Min:",
    np.min(latencies),
    "ms"
)

print(
    "Max:",
    np.max(latencies),
    "ms"
)

total_seconds = np.sum(latencies) / 1000

print(
    "Approx throughput:",
    NUM_REQUESTS / total_seconds,
    "requests/sec"
)