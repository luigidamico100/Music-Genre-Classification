import torch
import logging
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

# Model training
train_debug_mode = 'True'

batch_size = 128
epochs = 100
learning_rate = 0.001

sample_rate = 22050
# num_samples = 661794  # most frequent number of samples per song
# num_samples = sample_rate * 29  # 29 sec of song
chunks_len_sec = 14.

def set_logger(path_logger):

    logging.basicConfig(level=logging.INFO,
                        # format="%(message)s",
                        format="%(message)s",
                        handlers=[
                            logging.FileHandler(path_logger),
                            logging.StreamHandler(sys.stdout)
                            ]
                        )


