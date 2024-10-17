# Load the pretrained googlenet onnx file
#

import onnxruntime as ort

# Load ONNX model
model_path = '../presetFreeImaging_googlenet.onnx'
session = ort.InferenceSession(model_path)


# Get model input information
print("Input Details:")
for input_tensor in session.get_inputs():
    print(f"Name: {input_tensor.name}, Shape: {input_tensor.shape}, Type: {input_tensor.type}")

# Get model output information
print("\nOutput Details:")
for output_tensor in session.get_outputs():
    print(f"Name: {output_tensor.name}, Shape: {output_tensor.shape}, Type: {output_tensor.type}")
