import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomDecoder(nn.Module):
    def __init__(self, latent_dim, image_size):
        super(RandomDecoder, self).__init__()
        self.image_size = image_size
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 3 * self.image_size * self.image_size)  # RGB image (3 channels)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 3, self.image_size, self.image_size)  # Reshape into image format
        return torch.tanh(x)  # Normalize the output to [-1, 1] for image