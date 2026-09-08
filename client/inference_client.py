import numpy as np
import tritonclient.http as httpclient


TRITON_URL = "localhost:8000"
MODEL_NAME = "classifier"
MODEL_VERSION = "1"


# --------------------------------------------------
# Connect to Triton
# --------------------------------------------------

client = httpclient.InferenceServerClient(
    url=TRITON_URL
)


# --------------------------------------------------
# Health checks
# --------------------------------------------------

print("Server ready:", client.is_server_ready())

print(
    "Model ready:",
    client.is_model_ready(
        MODEL_NAME,
        MODEL_VERSION
    )
)


# --------------------------------------------------
# Create model input
# --------------------------------------------------

input_data = np.array(
    [[0.1, 0.2, 0.3, 0.4]],
    dtype=np.float32
)


# --------------------------------------------------
# Describe input tensor
# --------------------------------------------------

input_tensor = httpclient.InferInput(
    "input",
    input_data.shape,
    "FP32"
)

input_tensor.set_data_from_numpy(input_data)


# --------------------------------------------------
# Tell Triton which output we want
# --------------------------------------------------

requested_output = httpclient.InferRequestedOutput(
    "output"
)


# --------------------------------------------------
# Send inference request
# --------------------------------------------------

response = client.infer(
    model_name=MODEL_NAME,
    model_version=MODEL_VERSION,
    inputs=[input_tensor],
    outputs=[requested_output]
)


# --------------------------------------------------
# Convert Triton response → NumPy
# --------------------------------------------------

output = response.as_numpy("output")


print("\nInput:")
print(input_data)

print("\nPrediction:")
print(output)

print("\nPrediction shape:")
print(output.shape)