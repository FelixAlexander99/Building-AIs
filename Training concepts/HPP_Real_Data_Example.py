import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 1. ARCHITECTURE ---
class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# --- 2. DATASET ---
class HouseDataset(Dataset):
    def __init__(self):
        self.data = [
            # Size, Rooms,  Price
            [2000, 3,       400], 
            [1500, 2,       300],
            [3500, 5,       700],
            [1200, 1,       250],
            [2500, 4,       500],
            [4000, 6,       850],
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        features = row[:2] 
        price = row[2]
        
        # --- FIX #1: SCALING ---
        # We normalize manually here to keep numbers small.
        # Size / 1000, Rooms / 1 (leave as is)
        features[0] = features[0] / 1000.0 
        # Price / 100 (So 400 becomes 4.0)
        price = price / 100.0
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        price_tensor = torch.tensor([price], dtype=torch.float32)
        return features_tensor, price_tensor

# --- 3. SETUP ---
model = HousePriceModel()
dataset = HouseDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

criterion = nn.MSELoss()
# Since we scaled the data, a standard Learning Rate works fine now
optimizer = optim.SGD(model.parameters(), lr=0.01) 

# --- 4. TRAINING LOOP ---
print("--- Starting Training ---")

for epoch in range(1001): 
    
    total_loss = 0 # Track loss for this epoch
    
    for batch_inputs, batch_targets in dataloader:
        # Standard Loop
        optimizer.zero_grad()
        predictions = model(batch_inputs)
        loss = criterion(predictions, batch_targets)
        loss.backward()
        optimizer.step()
        
        # Accumulate loss (just for printing)
        total_loss += loss.item()

    # --- FIX #2: CLEANER PRINTING ---
    # Print average loss only once per epoch
    if epoch % 50 == 0:
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")


# --- 5. FINAL EVALUATION (Phase 6 Preview) ---
print("\n--- Final Results (Denormalized) ---")

# We turn off gradient tracking for evaluation (saves memory)
with torch.no_grad():
    # Let's check the whole dataset at once
    # We create a new loader that grabs EVERYTHING (batch_size=6)
    eval_loader = DataLoader(dataset, batch_size=6, shuffle=False)
    
    for inputs, targets in eval_loader:
        preds = model(inputs)
        
        # Convert back to Python lists
        # We multiply by 100 to get the real price back (undoing our scaling)
        real_preds = [p * 100 for p in preds.squeeze().tolist()]
        real_targets = [t * 100 for t in targets.squeeze().tolist()]
        
        for i in range(len(real_preds)):
            print(f"House {i+1}: Predicted ${real_preds[i]:.0f}k | Actual ${real_targets[i]:.0f}k")