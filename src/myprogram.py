#!/usr/bin/env python
import os
import json
import string
import random
import torch
import torch.nn as nn
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class CharLSTM(nn.Module):
    """Simple character-level LSTM for next character prediction."""
    
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        # x: (batch, seq_len) of character indices
        embed = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        if hidden is None:
            output, hidden = self.lstm(embed)
        else:
            output, hidden = self.lstm(embed, hidden)
        
        # Use last output for prediction
        logits = self.fc(output[:, -1, :])  # (batch, vocab_size)
        return logits, hidden
    
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


class MyModel:
    """
    Character-level LSTM language model for next character prediction.
    """

    def __init__(self):
        # Character vocabulary - common ASCII + some Unicode
        self.chars = list(string.printable[:95])  # printable ASCII without whitespace control chars
        self.chars.extend(['\n', '\t'])
        
        # Add start/padding token
        self.pad_token = '<PAD>'
        self.chars.insert(0, self.pad_token)
        
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        # Model parameters
        self.embed_dim = 64
        self.hidden_dim = 128
        self.num_layers = 1
        self.context_len = 32  # How much context to use
        
        # Model will be initialized during training or loading
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False  # Track if model was actually trained
        
        # Default character frequencies (fallback when model not trained)
        self.default_chars = [' ', 'e', 't', 'a', 'o', 'i', 'n', 's', 'h', 'r']

    def _encode(self, text):
        """Convert text to tensor of character indices."""
        indices = []
        for c in text:
            if c in self.char_to_idx:
                indices.append(self.char_to_idx[c])
            else:
                # Unknown char -> use space as fallback
                indices.append(self.char_to_idx.get(' ', 1))
        return indices

    def _init_model(self):
        """Initialize the LSTM model."""
        self.model = CharLSTM(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        ).to(self.device)

    @classmethod
    def load_training_data(cls):
        """Load training data from files."""
        data = []
        
        # Check for training data in data directory
        data_paths = ['data/train.txt', 'data/corpus.txt']
        for path in data_paths:
            if os.path.exists(path):
                print(f"Loading training data from {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    data.extend(f.readlines())
        
        print(f"Loaded {len(data)} lines of training data")
        return data

    @classmethod
    def load_test_data(cls, fname):
        """Load test data - each line is a prefix to predict next char for."""
        data = []
        with open(fname, encoding='utf-8') as f:
            for line in f:
                inp = line[:-1] if line.endswith('\n') else line
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        """Write predictions to file."""
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir, epochs=5, lr=0.001):
        """Train the LSTM model on text data."""
        if not data:
            print('No training data provided, using default model')
            self._init_model()
            self.is_trained = False
            return
        
        self._init_model()
        self.model.train()
        self.is_trained = True
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Prepare training sequences
        all_text = ''.join(data)
        sequences = []
        
        # Create training examples: (context, next_char)
        for i in range(len(all_text) - 1):
            start = max(0, i - self.context_len + 1)
            context = all_text[start:i+1]
            next_char = all_text[i+1]
            sequences.append((context, next_char))
        
        print(f'Training on {len(sequences)} sequences')
        
        # Training loop
        batch_size = 64
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            # Shuffle sequences
            random.shuffle(sequences)
            
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                
                # Prepare batch
                max_len = max(len(ctx) for ctx, _ in batch)
                contexts = []
                targets = []
                
                for ctx, next_char in batch:
                    # Pad context to max_len
                    padded = [0] * (max_len - len(ctx)) + self._encode(ctx)
                    contexts.append(padded)
                    
                    target_idx = self.char_to_idx.get(next_char, self.char_to_idx.get(' ', 1))
                    targets.append(target_idx)
                
                contexts = torch.tensor(contexts, dtype=torch.long).to(self.device)
                targets = torch.tensor(targets, dtype=torch.long).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                logits, _ = self.model(contexts)
                loss = criterion(logits, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / max(n_batches, 1)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

    def run_pred(self, data):
        """Generate predictions for test data."""
        preds = []
        
        if self.model is None or not self.is_trained:
            # No model loaded or not trained, use default predictions
            print('Warning: Model not trained, using default character frequencies')
            for _ in data:
                preds.append(''.join(self.default_chars[:3]))
            return preds
        
        self.model.eval()
        
        with torch.no_grad():
            for inp in data:
                try:
                    # Prepare input
                    if inp:
                        context = inp[-self.context_len:]  # Use last context_len chars
                        encoded = self._encode(context)
                    else:
                        encoded = [0]  # Just padding for empty input
                    
                    x = torch.tensor([encoded], dtype=torch.long).to(self.device)
                    
                    # Get predictions
                    logits, _ = self.model(x)
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Get top 3 predictions
                    top_probs, top_indices = torch.topk(probs[0], k=min(10, self.vocab_size))
                    
                    top_chars = []
                    for idx in top_indices.tolist():
                        char = self.idx_to_char.get(idx, ' ')
                        # Skip padding token
                        if char != self.pad_token and len(top_chars) < 3:
                            top_chars.append(char)
                    
                    # Ensure we have 3 predictions
                    while len(top_chars) < 3:
                        for c in self.default_chars:
                            if c not in top_chars:
                                top_chars.append(c)
                                break
                    
                    preds.append(''.join(top_chars[:3]))
                    
                except Exception as e:
                    # Fallback on error
                    print(f'Warning: Error predicting for "{inp[:20]}...": {e}')
                    preds.append(''.join(self.default_chars[:3]))
        
        return preds

    def save(self, work_dir):
        """Save model to checkpoint file."""
        checkpoint = {
            'chars': self.chars,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'context_len': self.context_len,
            'is_trained': self.is_trained,
        }
        
        # Save config
        config_path = os.path.join(work_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f)
        
        # Save model weights if model exists
        if self.model is not None:
            model_path = os.path.join(work_dir, 'model.pt')
            torch.save(self.model.state_dict(), model_path)
            print(f'Saved model to {model_path}')
        
        # Also save a dummy checkpoint for compatibility
        with open(os.path.join(work_dir, 'model.checkpoint'), 'w') as f:
            f.write('lstm_model')

    @classmethod
    def load(cls, work_dir):
        """Load model from checkpoint file."""
        model = cls()
        
        # Load config
        config_path = os.path.join(work_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            model.chars = config.get('chars', model.chars)
            model.char_to_idx = {c: i for i, c in enumerate(model.chars)}
            model.idx_to_char = {i: c for i, c in enumerate(model.chars)}
            model.vocab_size = len(model.chars)
            model.embed_dim = config.get('embed_dim', model.embed_dim)
            model.hidden_dim = config.get('hidden_dim', model.hidden_dim)
            model.num_layers = config.get('num_layers', model.num_layers)
            model.context_len = config.get('context_len', model.context_len)
            model.is_trained = config.get('is_trained', False)
        
        # Load model weights
        model_path = os.path.join(work_dir, 'model.pt')
        if os.path.exists(model_path):
            model._init_model()
            model.model.load_state_dict(torch.load(model_path, map_location=model.device))
            model.model.eval()
            print(f'Loaded model from {model_path}')
        else:
            print('No model weights found, using default predictions')
        
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = MyModel()
        print(f'Using device: {model.device}')
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print(f'Training on {len(train_data)} samples')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print(f'Using device: {model.device}')
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
