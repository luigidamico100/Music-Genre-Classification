import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")


experiment_name = 'trial'

# Training params
train_debug_mode = 'True'
batch_size = 128
epochs = 100
learning_rate = 0.001

# Signal processing params
sample_rate = 22050
chunks_len_sec = 14.
# num_samples = 661794  # most frequent number of samples per song
# num_samples = sample_rate * 29  # 29 sec of song

# Mel spectrogram params
melspec_fft = 1024
melspec_hop_length = 512
melspec_n_mels = 64

    
class MyLogger:
    
    def __init__(self):
        self.text = ''
        
    def write(self, text):
        print(text)
        self.text += text + '\n\n'
        
    def write_on_file(self, filename):
        with open(filename, 'w') as f:
            f.write(self.text)
            





