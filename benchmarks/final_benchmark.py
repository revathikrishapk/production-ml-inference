import csv
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
    client = httpclient.InferenceServerClient(url=TRITON_URL)

    input_data = np.random.randn(1, 4).astype(np.float32)

    input_tensor = httpclient.InferInput(
        "input",
        input_data.shape,
        "FP32"
    )

    input_tensor.set_data_from_numpy(input_data)

    output_tensor = httpclient.InferRequestedOutput("output")

    start = time.perf_counter()

    client.infer(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        inputs=[input_tensor],
        outputs=[output_tensor]
    )

    end = time.perf_counter()

    return (end - start) * 1000


def run_benchmark():

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        latencies = list(
            executor.map(
                lambda _: send_request(),
                range(NUM_REQUESTS)
            )
        )

    total_time = time.perf_counter() - start

    latencies = np.array(latencies)

    results = {
        "requests": NUM_REQUESTS,
        "concurrency": CONCURRENCY,
        "mean_ms": np.mean(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "max_ms": np.max(latencies),
        "throughput_req_s": NUM_REQUESTS / total_time,
    }

    return results


def save_results(results):

    with open(
        "results/final_benchmark.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results.keys()
        )

        writer.writeheader()
        writer.writerow(results)


if __name__ == "__main__":

    results = run_benchmark()

    print("\n=== Final Benchmark ===")

    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    save_results(results)

    print("\nSaved to results/final_benchmark.csv")