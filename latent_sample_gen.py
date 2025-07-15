import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Assumes VAEWithTransformerDecoder is defined and trained, and DataLoaders exist

def freeze_vae_components(model):
    """
    Freeze encoder, GT processor, and mu/logvar heads. Leave decoder frozen as well if desired.
    """
    for param in model.encoder.parameters():
        param.requires_grad = False
    for layer in model.gt_layers:
        for param in layer.parameters():
            param.requires_grad = False
    for param in model.mu_head.parameters():
        param.requires_grad = False
    for param in model.logvar_head.parameters():
        param.requires_grad = False
    # Decoder can also be frozen if not used during latent extraction
    for param in model.decoder.parameters():
        param.requires_grad = False

class LatentDataset(Dataset):
    """
    Dataset of latent means mu(x) for each window x in original dataset.
    """
    def __init__(self, data_loader, vae_model, device='cpu'):
        self.latents = []
        self.device = device
        vae_model.to(device).eval()
        freeze_vae_components(vae_model)
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(device)
                # Forward through encoder and GT
                z = vae_model.encoder(batch)
                z = vae_model.process_latent(z)
                # Compute mu and ignore logvar
                mu = vae_model.mu_head(z)
                # Flatten per-sample latent matrix to vector
                # shape [B, T, D] -> [B, T*D]
                B, T, D = mu.shape
                mu_flat = mu.view(B, T * D).cpu().numpy()
                self.latents.append(mu_flat)
        self.latents = np.concatenate(self.latents, axis=0)

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.latents[idx]).float()


# Usage example:
# from latent_dataset_preparation import LatentDataset, freeze_vae_components
#
# # Assuming train_loader defined and model loaded:
# model = VAEWithTransformerDecoder(...)
# model.load_state_dict(torch.load('best_model.pt'))
#
# # Prepare latent dataset
# latent_dataset = LatentDataset(train_loader, model, device='cuda')
# latent_loader = DataLoader(latent_dataset, batch_size=64, shuffle=True)
#
# # Save latent array for diffusion training
# np.save('latents_train.npy', latent_dataset.latents)
