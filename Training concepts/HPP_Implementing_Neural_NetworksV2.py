import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. SETUP (The Architecture) ---
class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)   #Weights
        self.activation = nn.ReLU()     #Switch (- to 0s)
        self.layer2 = nn.Linear(4, 1)   #Output

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Initialize Model and Data
model = HousePriceModel()
inputs = torch.tensor([[2000., 3.], [1500., 2.], [3500., 5.]])
targets = torch.tensor([[400.], [300.], [700.]])

# --- 2. THE TOOLS (Loss & Optimizer) ---

# Loss Function: MSE (Mean Squared Error) - Standard for regression
criterion = nn.MSELoss()

# Optimizer: SGD (Stochastic Gradient Descent)
# lr=0.01 is the "Learning Rate" (Step size). 
# model.parameters() hands over the keys to the optimizer so it can access the weights.
optimizer = optim.SGD(model.parameters(), lr=0.001)


# --- 3. THE TRAINING LOOP ---
print(f"Initial Loss: {criterion(model(inputs), targets).item()}")

# We loop 1000 times (Epochs)
for epoch in range(1001):
    
    # STEP A: Forward Pass
    predictions = model(inputs)
    
    # STEP B: Calculate Loss
    loss = criterion(predictions, targets)
    
    # STEP C: Zero Gradients
    optimizer.zero_grad()
    
    # STEP D: Backward Pass
    loss.backward()
    
    # STEP E: Optimizer Step
    # Update the weights: weight -= grad * lr
    optimizer.step()
    
    # Monitoring
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item()}")


# --- 4. VERIFICATION ---
print("\n--- Final Results ---")
print(f"Target Prices: {targets.squeeze().tolist()}")
print(f"Predicted Prices: {model(inputs).detach().squeeze().tolist()}")