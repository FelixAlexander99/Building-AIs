import torch
import torch.nn as nn  # The library for building layers

# 1. THE ARCHITECTURE CLASS
# Every model in PyTorch inherits from nn.Module
class HousePriceModel(nn.Module):
    
    def __init__(self):
        """
        Listing the layers we want to use.
        We don't connect them yet.
        """
        super().__init__()
        
        # Layer 1 (The "Hidden" Layer)
        # Inputs: 2 (Size, Rooms)
        # Outputs: 4 (We invent 4 'neurons' to learn intermediate patterns)
        # This replaces: weights = torch.randn(2, 4)
        self.layer1 = nn.Linear(in_features=2, out_features=4)
        
        # The Activation Function (The "Switch")
        # ReLU (Rectified Linear Unit) turns negatives to zero.
        # It adds the "non-linearity".
        self.activation = nn.ReLU()
        
        # Layer 2 (The Output Layer)
        # Inputs: 4 (Must match the output of Layer 1)
        # Outputs: 1 (The Final Price)
        self.layer2 = nn.Linear(in_features=4, out_features=1)

    def forward(self, x):
        """
        In Forward, we define the flow of data.
        x is our input data.
        """
        # 1. Pass data through the first layer (Math)
        x = self.layer1(x)
        
        # 2. Pass through the activation (Non-Linearity)
        # This allows the model to be 'smart' and not just a linear equation.
        x = self.activation(x)
        
        # 3. Pass through final layer to get the price
        x = self.layer2(x)
        
        return x

# --- USING THE NEW MODEL ---

# 1. Create an instance of the model
# This automatically initializes all those random weights inside layer1 and layer2
model = HousePriceModel()

print("--- The Model Architecture ---")
print(model)

# 2. The Data (Same as before)
inputs = torch.tensor([[2000., 3.], [1500., 2.], [3500., 5.]])

# 3. The Forward Pass
# The parent class (nn.Module) has a pre-written __call__ method built in (callable object)
# "forward" is present in nn, so it gets overwritten in HousePriceModel class (method overriding)
# Calling it like this instead of model.forward(inputs) is preferable as this way it doesn't skip safety hooks
predictions = model(inputs)

print("\n--- Predictions (Randomized) ---")
print(predictions)

# Print the weights of the first layer
print("\n--- Weights ---")
print(model.layer1.weight)
print("\n--- BIAS ---")
print(model.layer1.bias)