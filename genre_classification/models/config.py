import torch
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

# ---------------------------- Editable params: -----------------------------#

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
melspec_hop_length = 256 #512
melspec_n_mels = 128 #64

# Evaluation params
set_ = 'val'

# --------------------------------------------------------------------------- #


# Reproducibility
random_seed = 42
import torch
torch.manual_seed(random_seed)
import random
random.seed(random_seed)
import numpy as np
np.random.seed(random_seed)

    
class MyLogger:
    
    def __init__(self):
        self.text = ''
        
    def write(self, text):
        print(text)
        self.text += text + '\n\n'
        
    def write_on_file(self, filename):
        with open(filename, 'w') as f:
            f.write(self.text)
            

def parse_params(config, reason='training'):
    
    assert reason in ['training', 'evaluate']
    
    params = {}
    
    if reason=='training':
    
        parser = argparse.ArgumentParser(description='Training process')
        parser.add_argument('--epochs', type=int, help='epochs number', default=config.epochs)
        # parser.add_argument('--train_debug_mode', type=bool, help='Train debug mode', default=train_debug_mode, action=argparse.BooleanOptionalAction)
        parser.add_argument('--train_debug_mode', type=str, help='Train debug mode', default=config.train_debug_mode)
        parser.add_argument('--batch_size', type=int, help='batch size', default=config.batch_size)
        parser.add_argument('--learning_rate', type=float, help='training learning rate', default=config.learning_rate)
        parser.add_argument('--experiment_name', type=str, help='experiment name', default=config.experiment_name)
        parser.add_argument('--chunks_len_sec', type=float, help='Chunks length (sec)', default=config.chunks_len_sec)
        parser.add_argument('--melspec_fft', type=int, help='melspec fft', default=config.melspec_fft)
        parser.add_argument('--melspec_hop_length', type=float, help='melspec hop length', default=config.melspec_hop_length)
        parser.add_argument('--melspec_n_mels', type=float, help='melspec n mels', default=config.melspec_n_mels)
        
        args = parser.parse_args()
        train_debug_mode = args.train_debug_mode == 'True'
        learning_rate = args.learning_rate
        batch_size = args.batch_size
        experiment_name = args.experiment_name
        chunks_len_sec = args.chunks_len_sec
        n_examples = 'all' if not train_debug_mode else 50
        epochs = args.epochs if not train_debug_mode else 3
        melspec_fft = args.melspec_fft
        melspec_hop_length = args.melspec_hop_length
        melspec_n_mels = args.melspec_n_mels
        
        params['train_debug_mode'] = train_debug_mode
        params['n_examples'] = n_examples
        params['experiment_name'] = experiment_name
        params['epochs'] = epochs
        params['batch_size'] = batch_size
        params['learning_rate'] = learning_rate
        params['chunks_len_sec'] = chunks_len_sec
        params['melspec_fft'] = melspec_fft
        params['melspec_hop_length'] = melspec_hop_length
        params['melspec_n_mels'] = melspec_n_mels
        
        return params
    
    elif reason=='evaluate':
        parser = argparse.ArgumentParser(description='Evaluate process')
        parser.add_argument('--experiment_name', type=str, help='Experiment name', default=config.experiment_name)
        parser.add_argument('--set', type=str, help='Choose from (all, train, val, test)', default=config.set_)
        args = parser.parse_args()
        
        params['experiment_name'] = args.experiment_name
        params['set'] = args.set
        
        
        return params




