#Imports
import torch

#Sanity checks
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

# --- STEP 1: Creating the "Input" Tensor (Batch of Data) ---
# Let's pretend we have 3 houses.
# Each house has 3 features: [Size (sq ft), Number of Rooms, stories]
# We use standard lists first to show how they convert to tensors.
data_raw = [
    [2000, 3, 3], # House 1
    [1500, 2, 2], # House 2
    [3500, 5, 1]  # House 3
]

# Convert to a PyTorch Tensor
# float() is important because PyTorch does math in floating point precision
# This uses scientific notation for the purposes of readability
inputs = torch.tensor(data_raw).float() 

print("--- INPUTS ---")
print(inputs)
print(f"Shape: {inputs.shape}") # Expecting [3, 2] -> [3 rows/samples, 2 cols/features]

# --- STEP 2: Creating the "Weights" Tensor ---
# We need 3 weights: "Size" / "Rooms" / "Stories"
# Nr columns can be anything in this example
weights = torch.randn(3, 1) 

print("\n--- WEIGHTS (Randomized) ---")
print(weights)
print(f"Shape: {weights.shape}")


# --- STEP 3: The "Bias" Tensor ---
# The base price of a house regardless of size (a scalar).
bias = torch.tensor([5000.0])

print("\n--- BIAS ---")
print(bias)


# --- STEP 4: The Math (Matrix Multiplication & Broadcasting) ---
# (Inputs x Weights) + Bias
# mm stands for "Matrix Multiplication"
predictions = torch.mm(inputs, weights) + bias

print("\n--- PREDICTIONS ---")
print(predictions)
print(f"Shape: {predictions.shape}") 
# Notice the shape is [3, 1]. We predicted 3 prices, one for each house.