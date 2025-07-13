import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    Simple MLP for classification based on latent z.
    """
    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, num_classes)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            logits: (batch, num_classes)
        """
        return self.classifier(z)


def train_classifier(
    vae: nn.Module,
    classifier: ClassificationHead,
    dataset: Dataset,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4
):
    """
    Train a classification head on latent representations from the VAE.

    Args:
        vae: pretrained TransformerVAE encoder
        classifier: ClassificationHead instance
        dataset: Dataset yielding (x, y)
        epochs, batch_size, lr: training hyperparameters
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.to(device).eval()
    classifier.to(device).train()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                mu, logvar = vae.encode(x)
                z = vae.reparameterize(mu, logvar)
            logits = classifier(z)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
        print(f"[Classifier Train] Epoch {epoch+1}/{epochs}, Loss: {total_loss/total:.4f}, Acc: {correct/total:.4f}")
