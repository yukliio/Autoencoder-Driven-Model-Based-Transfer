import torch
import numpy as np
import sys
sys.path.append('..')
from models.autoencoder import TrajectoryAutoEncoder

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TrajectoryAutoEncoder(feature_dim=61, hidden_size=256, latent_dim=32, seq_len=305).to(device)
model.load_state_dict(torch.load('../checkpoints/autoencoder_best.pth'))
model.eval()

# Load processed data
data = np.load('../data/processed/intersectionzoo_latent_mbtldata.npy')
data_tensor = torch.FloatTensor(data).to(device)

# Extract latent vectors
with torch.no_grad():
    _, latents = model(data_tensor)
    latents = latents.cpu().numpy()

# Save latent vectors
np.save('../data/processed/latent_vectors.npy', latents)
print(f'Extracted {len(latents)} latent vectors of dimension {latents.shape[1]}')
print('Saved to data/processed/latent_vectors.npy')