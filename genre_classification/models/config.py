import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

# Model training

batch_size = 128
epochs = 100
learning_rate = 0.001

sample_rate = 22050
num_samples = 661794
