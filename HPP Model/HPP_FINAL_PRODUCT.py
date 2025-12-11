import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os

input_size = 10

# --- 1. ARCHITECTURE ---
class HousePriceModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # Hidden Layer 1
        self.layer1 = nn.Linear(input_size, 128)
        # LeakyReLU: negative_slope=0.01 is default, 
        # but 0.1 allows a bit more signal through for dead neurons.
        self.act1 = nn.LeakyReLU(negative_slope=0.1)
        self.drop1 = nn.Dropout(0.2) # 20% chance to drop a neuron
        
        # Hidden Layer 2: Added depth to learn complex combinations
        self.layer2 = nn.Linear(128, 64)
        self.act2 = nn.LeakyReLU(negative_slope=0.1)
        self.drop2 = nn.Dropout(0.2)

        # Hidden Layer 3: Refine logic
        self.layer3 = nn.Linear(64,32)
        self.act3 = nn.LeakyReLU(negative_slope=0.1)
        
        # Output Layer
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.drop1(x) # Apply dropout during training only
        
        x = self.act2(self.layer2(x))
        x = self.drop2(x)
        
        x = self.act3(self.layer3(x))
        
        x = self.output(x)
        return x

# --- 2. DATASET (Expanded) ---
class HouseDataset(Dataset):
    def __init__(self, csv_file):
        # Load raw data
        df = pd.read_csv(csv_file)

        # Select features and target
        columns_to_drop = ['longitude', 'latitude', 'housing_median_age']
        df = df.drop(columns_to_drop, axis = 1)

        # Drop any bits of missing data
        df = df.dropna()

        # ONE-HOT encoding
        # Take ocean_proximity column and turn it into 1s and 0s
        df = pd.get_dummies(df, columns=['ocean_proximity'], dtype = float)
        # Current df has: total_rooms, total_bedrooms, population, households, median_income, median_house_value

        # Separate targets from features
        self.targets = df.pop('median_house_value').values #.pop() removes the column from df and returns it
        features = df.values

        # Calculate statistics (for normalization)
        features_mean = features.mean(axis=0)
        features_std = features.std(axis=0)

        #NORMALIZE once before working on the data
        self.features = (features - features_mean) / (features_std + 1e-6)

        # Most target values are 6 digits
        self.target_scale = 100000.0
        self.targets = self.targets / self.target_scale

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Create tensors
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor([self.targets[idx]], dtype=torch.float32)
        
        return features, target
    
# For Windows specifically (no automatic forking)
# Necessary for increasing num_workers
if __name__ == '__main__':
    # Automatically pick the best device: CUDA (Nvidia), MPS (Mac), or CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    print(f"Using device: {device}")

    # Check for correct file path
    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dataset_dir, "HP_California.csv")
    print(f"Looking for file at: {file_path}")


    # --- 3. PREPARATION ---
    # A. Initialize Data
    full_dataset = HouseDataset(file_path)
    sample_input, sample_target = full_dataset[0]
    print(f"Input Features: {sample_input.shape[0]}")

    # UPGRADE 2: TRAIN / VAL SPLIT
    # 80% for Training (80 items), 20% for Validation (20 items)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # B. Loaders
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True) # No need to shuffle val

    # C. Model Setup
    input_features = full_dataset[0][0].shape[0]
    model = HousePriceModel(input_features).to(device) # Move model to GPU
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Switch to Adam (usually better/faster)


    # --- 4. THE TRAINING & VALIDATION LOOP ---
    print("--- Starting Training ---")

    best_val_loss = float('inf') # To track the best model

    for epoch in range(300):
        
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
    print("\n--- Loading Best Model for Real Usage ---")

    final_model = HousePriceModel(input_size).to(device)
    final_model.load_state_dict(torch.load('best_house_model.pth', weights_only=True))
    final_model.eval()

    # Using a real sample from validation set (which has all 10 features correctly formatted)
    # val_dataset[0] returns (input_tensor, target_tensor)
    test_input, test_target = val_dataset[0]
    
    # Adding a "Batch Dimension"
    test_input = test_input.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = final_model(test_input)
        
        # NORMALIZATION (using dataset scaling)
        predicted_price = prediction.item() * 100000.0
        actual_price = test_target.item() * 100000.0

    print(f"Test House Prediction:")
    print(f"Predicted Price: ${predicted_price:,.2f}")
    print(f"Actual Price:    ${actual_price:,.2f}")
    
    difference = abs(predicted_price - actual_price)
    print(f"Difference:      ${difference:,.2f}")