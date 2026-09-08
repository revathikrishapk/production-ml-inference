\# Production ML Inference Platform



A production-style machine learning inference platform built with FastAPI, NVIDIA Triton Inference Server, ONNX Runtime, Docker Compose, and Prometheus.



The project focuses on reliable model serving, dynamic batching, health checks, failure handling, observability, and measured inference performance.



\---



\## Architecture



```text

&#x20;                        Client

&#x20;                          |

&#x20;                          v

&#x20;                   +--------------+

&#x20;                   |   FastAPI    |

&#x20;                   |    :8080     |

&#x20;                   +------+-------+

&#x20;                          |

&#x20;                          v

&#x20;                 +-------------------+

&#x20;                 | NVIDIA Triton     |

&#x20;                 |     :8000         |

&#x20;                 | Dynamic Batching  |

&#x20;                 +---------+---------+

&#x20;                           |

&#x20;                           v

&#x20;                   +---------------+

&#x20;                   | ONNX Runtime  |

&#x20;                   |  Classifier    |

&#x20;                   +---------------+



&#x20;            +---------------------------+

&#x20;            |       Prometheus          |

&#x20;            |          :9090             |

&#x20;            +-------------+-------------+

&#x20;                          |

&#x20;                   +------+------+

&#x20;                   |             |

&#x20;                   v             v

&#x20;              FastAPI         Triton

&#x20;              /metrics        /metrics



text```



\### Tech stack

Python

PyTorch

ONNX

ONNX Runtime

NVIDIA Triton Inference Server

FastAPI

Docker

Docker Compose

Prometheus

NumPy



\### 

Key Features

Model Serving



The trained model is exported to ONNX and served through NVIDIA Triton Inference Server using the ONNX Runtime backend.



Dynamic Batching



Triton's dynamic batching is enabled to combine incoming inference requests into larger batches when possible.



API Layer



FastAPI provides:



GET /health

GET /ready

POST /predict

GET /metrics/

Health and Readiness



The API distinguishes between:



application health

model/server readiness

dependency failure



If Triton becomes unavailable, the API returns:



503 Service Unavailable



instead of exposing an unexpected server error.



Observability



Prometheus collects metrics from both FastAPI and Triton.



FastAPI metrics include:



request count

HTTP status codes

request latency



Triton metrics include:



inference count

execution count

inference failures

queue duration

inference duration

pending requests

Performance

Triton Dynamic Batching A/B Test



Test configuration:



Requests: 100

Concurrency: 8

Configuration	Throughput	P50	P95	P99

Dynamic batching OFF	349.0 req/s	16.65 ms	33.86 ms	39.11 ms

Dynamic batching ON	416.0 req/s	14.32 ms	28.82 ms	36.88 ms

Result



Dynamic batching produced:



19.2% higher throughput

14.9% lower p95 latency



The experiment demonstrates the effect of request batching under concurrent inference traffic.



End-to-End API Benchmark



The complete FastAPI → Triton → ONNX Runtime path was also benchmarked.



Configuration:



Requests: 100

Concurrency: 8



Results:



Metric	Result

Mean latency	21.08 ms

P50 latency	15.69 ms

P95 latency	79.14 ms

P99 latency	86.61 ms

Maximum latency	92.50 ms

Throughput	122.62 req/s



These numbers represent the complete API path rather than Triton's internal execution alone.



Reliability Testing



The system was tested under both normal and failure conditions.



Scenario	Expected	Result

Valid prediction	200	PASS

Invalid feature count	400	PASS

FastAPI health	200	PASS

Triton ready	200	PASS

Triton unavailable	503	PASS

Triton recovery	200	PASS



Example:



Triton unavailable

&#x20;      |

&#x20;      v

FastAPI /ready

&#x20;      |

&#x20;      v

503 Service Unavailable



After Triton recovery:



Triton healthy

&#x20;      |

&#x20;      v

FastAPI /ready

&#x20;      |

&#x20;      v

200 OK

API Usage

Health

curl http://localhost:8080/health



Response:



{

&#x20; "status": "healthy"

}

Readiness

curl http://localhost:8080/ready



Response:



{

&#x20; "status": "ready",

&#x20; "model": "classifier",

&#x20; "version": "1"

}

Prediction

curl -X POST http://localhost:8080/predict \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{"features":\[0.1,0.2,0.3,0.4]}'

Metrics

http://localhost:8080/metrics/



Prometheus:



http://localhost:9090



Triton metrics:



http://localhost:8002/metrics

Running the Project

1\. Start the stack

docker compose up -d

2\. Check containers

docker compose ps



Expected:



ml-api          Up

triton-server   Up (healthy)

prometheus      Up

3\. Test inference

curl http://localhost:8080/health

curl http://localhost:8080/ready

curl -X POST http://localhost:8080/predict \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{"features":\[0.1,0.2,0.3,0.4]}'

4\. Stop the stack

docker compose down

Project Structure

production-ml-inference/

│

├── app/

│   ├── \_\_init\_\_.py

│   └── main.py

│

├── model\_repository/

│   └── classifier/

│       ├── config.pbtxt

│       └── 1/

│           └── model.onnx

│

├── client/

│   └── inference\_client.py

│

├── benchmarks/

│   ├── baseline.py

│   ├── concurrency\_baseline.py

│   └── final\_benchmark.py

│

├── prometheus/

│   └── prometheus.yml

│

├── results/

│   ├── comparison.csv

│   └── final\_benchmark.csv

│

├── Dockerfile

├── docker-compose.yml

├── requirements.txt

└── README.md

Engineering Highlights

Exported a machine learning model to ONNX for portable inference.

Deployed the model using NVIDIA Triton Inference Server.

Implemented dynamic batching for concurrent inference workloads.

Built a FastAPI inference gateway over Triton.

Containerized the complete inference stack with Docker Compose.

Added health and readiness checks for service reliability.

Implemented graceful 400 and 503 error handling.

Added Prometheus monitoring for API and model-serving metrics.

Measured a 19.2% throughput improvement and 14.9% p95 latency reduction from dynamic batching.

Validated recovery behavior after Triton failure.

