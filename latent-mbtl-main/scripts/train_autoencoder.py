import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import sys
sys.path.append('..')
from models.autoencoder import TrajectoryAutoEncoder


# Custom Dataset file
class TrajectoryDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.FloatTensor(np.load(data_path))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# Load data file
dataset = TrajectoryDataset('../data/processed/intersectionzoo_latent_mbtldata.npy')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TrajectoryAutoEncoder(feature_dim=61, hidden_size=256, latent_dim=32, seq_len=500).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"Using device: {device}")
print(f"Training on {train_size} samples, validating on {val_size} samples")

# Training loops
epochs = 100
best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon, z = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon, z = model(batch)
            loss = criterion(recon, batch)
            val_loss += loss.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # save model best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '../checkpoints/autoencoder_best.pth')
        print(f'  --> Saved best model')

print('Training complete!')