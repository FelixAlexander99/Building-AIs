#Imports
import torch

'''
Explanation so far:
We take our batch of data that needs work(inputs) and we know what values we're looking for(targets). 
The weights are the controller of sorts, we verify whether increasing or decreasing them gets us closer to the targets
The derivative represents whether we get closer or further away from those targets after changing the weight
A positive derivative means the error goes UP, so we should do the opposite of the latest weight change
A negative derivative means the error goes DOWN, so we should keep doing what we just did
A derivative of ZERO means we reached the target or the weights are broken
A gradient is a collection of derivatives, in our case the weights for SIZE, ROOMS & BIAS
'''

#Sanity checks
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

# --- STEP 1: Creating the "Input" Tensor (Batch of Data) ---
# Let's pretend we have 3 houses.
# Each house has 2 features: [Size (sq ft), Number of Rooms]
# This uses scientific notation for the purposes of readability
inputs = torch.tensor([[2000., 3.], [1500., 2.], [3500., 5.]])

print("--- INPUTS ---")
print(inputs)
print(f"Shape: {inputs.shape}") # Expecting [3, 2] -> [3 rows/samples, 2 cols/features]

# The "Correct" answers (The prices these houses actually sold for)
# Let's say: $400k, $300k, $700k
targets = torch.tensor([[400.], [300.], [700.]])

# --- STEP 2: Creating the "Weights" Tensor ---
# We need 2 weights: "Size" / "Rooms"

weights = torch.randn(2, 1, requires_grad=True) #3rd param "records" values
bias = torch.randn(1, requires_grad=True) #Base price for a house, taken at random

print("\n--- WEIGHTS (Randomized) ---")
print(weights)
print(f"Shape: {weights.shape}")
print(f"Initial Bias: {bias}")


# --- STEP 3. FORWARD PASS
# PyTorch is building a 'graph' in the background connecting these variables.
predictions = torch.mm(inputs, weights) + bias

# --- STEP 4. CALCULATE THE "LOSS" (The Error)
# How wrong am I? Uses Mean Squared Error (MSE).
# Math: Average of (Prediction - Target)^2
loss = (predictions - targets).pow(2).mean()

print(f"\nCurrent Loss (Error): {loss.item()}")

# --- STEP 5. BACKWARD PASS
# PyTorch calculates mean (average of 3 houses)
# ...how sensitive the average error was for each weight (derivative)
# ...stores all sensitivity values into the .grad attribute so we know how to change weights
loss.backward()

# --- STEP 6. INSPECT THE GRADIENTS
# PyTorch deposited the results into the .grad attribute of the weights.
print("\n--- THE GRADIENTS (The Instruction Manual) ---")
print("Gradients for Weights:")
print(weights.grad)

print("\nGradients for Bias:")
print(bias.grad)


# --- STEP 7. A simple manual update:
# New Weight = Old Weight - (Small Step * Gradient)
learning_rate = 0.0001 

# We wrap this in 'no_grad' because we don't want PyTorch to track this update step 
# as part of the math history (it gets recursive and messy otherwise).
with torch.no_grad():
    weights -= weights.grad * learning_rate
    bias -= bias.grad * learning_rate
    
    # Now, let's clear the gradients (reset them to zero) for the next round
    weights.grad.zero_()
    bias.grad.zero_()

# Check if our new weights are better!
new_predictions = torch.mm(inputs, weights) + bias
new_loss = (new_predictions - targets).pow(2).mean()

print(f"\nNew Loss after 1 step: {new_loss.item()}")
