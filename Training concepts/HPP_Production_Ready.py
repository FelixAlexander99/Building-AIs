import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import random # To generate more fake data

# --- UPGRADE 1: DEVICE SETUP ---
# automatically pick the best device: CUDA (Nvidia), MPS (Mac), or CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)
print(f"Using device: {device}")

# --- 1. ARCHITECTURE ---
class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 8) # Increased neurons slightly
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# --- 2. DATASET (Expanded) ---
class HouseDataset(Dataset):
    def __init__(self, num_samples=100):
        # We generate 100 fake houses so we have enough data to split
        self.data = []
        for _ in range(num_samples):
            size = random.randint(500, 5000)
            rooms = random.randint(1, 10)
            # Create a "secret formula" for price so the AI has something to learn
            # Price = Size * 0.2 + Rooms * 10 + Random Noise
            price = (size * 0.2) + (rooms * 10.0) + random.randint(-20, 20)
            self.data.append([size, rooms, price])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        
        # Normalize (Scale down inputs)
        size_norm = row[0] / 1000.0
        rooms_norm = row[1] / 10.0
        price_norm = row[2] / 100.0
        
        # Create tensors
        features = torch.tensor([size_norm, rooms_norm], dtype=torch.float32)
        target = torch.tensor([price_norm], dtype=torch.float32)
        
        return features, target

# --- 3. PREPARATION ---

# A. Initialize Data
full_dataset = HouseDataset(num_samples=100)

# UPGRADE 2: TRAIN / VAL SPLIT
# 80% for Training (80 items), 20% for Validation (20 items)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# B. Loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False) # No need to shuffle val

# C. Model Setup
model = HousePriceModel().to(device) # Move model to GPU
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01) # Switch to Adam (usually better/faster)


# --- 4. THE TRAINING & VALIDATION LOOP ---
print("--- Starting Training ---")

best_val_loss = float('inf') # To track the best model

for epoch in range(1000):
    
    # --- A. TRAINING PHASE ---
    model.train() # Turn on "Training Mode"
    train_loss = 0.0
    
    for inputs, targets in train_loader:
        # Move data to the same device as the model
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader)

    # --- B. VALIDATION PHASE ---
    # Check performance on data the model has NEVER seen
    model.eval() # Turn on "Evaluation Mode" (disables specific training features)
    val_loss = 0.0
    
    with torch.no_grad(): # Don't calculate gradients here (saves memory)
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    
    # --- C. LOGGING & SAVING ---
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # UPGRADE 3: CHECKPOINTING
        # If this model is better than the previous best, save it.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the "State Dict" (The dictionary of weights)
            torch.save(model.state_dict(), 'best_house_model.pth')
            print("  -> New best model saved!")

print("\n--- Training Complete ---")


# --- 5. LOADING AND INFERENCE ---
print("--- Loading Best Model for Real Usage ---")

# Create a fresh model instance (random weights)
final_model = HousePriceModel().to(device)

# Load the trained weights from the file
final_model.load_state_dict(torch.load('best_house_model.pth', weights_only=True))
final_model.eval()

# Let's predict a specific house: 2000 sq ft, 3 rooms
# Formula was: 2000*0.2 (400) + 3*10 (30) = 430 approx
manual_input = torch.tensor([[2000/1000.0, 3/10.0]]).float().to(device)

with torch.no_grad():
    prediction = final_model(manual_input)
    # Undo normalization (x100)
    real_price = prediction.item() * 100

print(f"House (2000 sqft, 3 rooms)")
print(f"Predicted Price: ${real_price:.2f}k")