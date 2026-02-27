#!/usr/bin/env python
import os
import json
import string
import random
import collections
import torch
import torch.nn as nn
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        if hidden is None:
            output, hidden = self.lstm(embed)
        else:
            output, hidden = self.lstm(embed, hidden)
        logits = self.fc(self.dropout(output[:, -1, :]))
        return logits, hidden

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


class MyModel:
    """
    Hybrid character-level prediction model:
      - Primary: n-gram lookup table (very fast, O(1) dict lookups)
      - Backup: LSTM neural network (handles unseen contexts)
    """

    def __init__(self):
        self.chars = list(string.printable[:95])
        self.chars.extend(['\n', '\t'])
        self.pad_token = '<PAD>'
        self.chars.insert(0, self.pad_token)

        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        self.embed_dim = 64
        self.hidden_dim = 128
        self.num_layers = 1
        self.context_len = 32

        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False

        # N-gram model: {order: {context_string: "top3chars"}}
        self.ngram_preds = {}
        self.max_ngram_order = 8

        # Unigram (character frequency) fallback
        self.unigram_top = [' ', 'e', 't', 'a', 'o', 'i', 'n', 's', 'h', 'r']

    def _encode(self, text):
        indices = []
        for c in text:
            if c in self.char_to_idx:
                indices.append(self.char_to_idx[c])
            else:
                indices.append(self.char_to_idx.get(' ', 1))
        return indices

    def _init_model(self):
        self.model = CharLSTM(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        ).to(self.device)

    # ── N-gram methods ────────────────────────────────────────────────

    def _build_ngram_model(self, text):
        """Build character n-gram prediction tables from training text."""
        print(f"Building n-gram model (orders 1-{self.max_ngram_order}) on {len(text):,} chars...")
        self.ngram_preds = {}

        freq = collections.Counter(text)
        self.unigram_top = [c for c, _ in freq.most_common(10)]

        for order in range(1, self.max_ngram_order + 1):
            counts = {}
            for i in range(len(text) - order):
                ctx = text[i:i + order]
                nxt = text[i + order]
                if ctx not in counts:
                    counts[ctx] = {}
                d = counts[ctx]
                d[nxt] = d.get(nxt, 0) + 1

            min_count = max(2, order)
            order_preds = {}
            for ctx, char_counts in counts.items():
                total = sum(char_counts.values())
                if total >= min_count:
                    sorted_chars = sorted(char_counts.items(), key=lambda x: -x[1])
                    order_preds[ctx] = ''.join(c for c, _ in sorted_chars[:5])
            self.ngram_preds[order] = order_preds
            print(f"  Order {order}: {len(order_preds):,} contexts")
            del counts

    def _predict_ngram(self, text):
        """Return top-3 prediction string from n-gram model, or None."""
        if not text:
            return None
        for order in range(self.max_ngram_order, 0, -1):
            preds = self.ngram_preds.get(order)
            if preds is None or len(text) < order:
                continue
            ctx = text[-order:]
            pred = preds.get(ctx)
            if pred and len(pred) >= 3:
                return pred[:3]
        return None

    # ── Data loading ──────────────────────────────────────────────────

    @classmethod
    def load_training_data(cls):
        data = []
        for path in ['data/train.txt', 'data/corpus.txt']:
            if os.path.exists(path):
                print(f"Loading training data from {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    data.extend(f.readlines())
        print(f"Loaded {len(data):,} lines of training data")
        return data

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname, encoding='utf-8') as f:
            for line in f:
                inp = line[:-1] if line.endswith('\n') else line
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    # ── Training ──────────────────────────────────────────────────────

    def run_train(self, data, work_dir, epochs=3, lr=0.002):
        if not data:
            print('No training data provided')
            self._init_model()
            return

        all_text = ''.join(data)
        text_len = len(all_text)
        print(f"Total training text: {text_len:,} characters")

        # 1) Build n-gram model
        self._build_ngram_model(all_text)

        # 2) Train LSTM (on sampled sequences - n-grams handle common cases)
        self._init_model()
        self.model.train()
        self.is_trained = True

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        seqs_per_epoch = min(50_000, text_len - 1)
        batch_size = 256
        print(f"LSTM training: {epochs} epochs, {seqs_per_epoch:,} sequences/epoch")

        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0

            # Sample random positions each epoch
            positions = random.sample(range(text_len - 1), seqs_per_epoch)
            random.shuffle(positions)

            for bi in range(0, len(positions), batch_size):
                batch_pos = positions[bi:bi + batch_size]
                max_ctx = 0
                raw = []
                for pos in batch_pos:
                    start = max(0, pos - self.context_len + 1)
                    ctx = all_text[start:pos + 1]
                    nxt = all_text[pos + 1]
                    raw.append((ctx, nxt))
                    if len(ctx) > max_ctx:
                        max_ctx = len(ctx)

                contexts = []
                targets = []
                for ctx, nc in raw:
                    padded = [0] * (max_ctx - len(ctx)) + self._encode(ctx)
                    contexts.append(padded)
                    targets.append(self.char_to_idx.get(nc, self.char_to_idx.get(' ', 1)))

                contexts_t = torch.tensor(contexts, dtype=torch.long).to(self.device)
                targets_t = torch.tensor(targets, dtype=torch.long).to(self.device)

                optimizer.zero_grad()
                logits, _ = self.model(contexts_t)
                loss = criterion(logits, targets_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)
            cur_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {cur_lr:.6f}')

    # ── Prediction ────────────────────────────────────────────────────

    def run_pred(self, data):
        preds = []
        ngram_hits = 0

        if self.model is not None:
            self.model.eval()

        with torch.no_grad():
            for inp in data:
                # Primary: n-gram lookup (fast)
                ngram_pred = self._predict_ngram(inp)
                if ngram_pred:
                    preds.append(ngram_pred)
                    ngram_hits += 1
                    continue

                # Backup: LSTM
                if self.model is not None and self.is_trained:
                    try:
                        context = inp[-self.context_len:] if inp else ''
                        encoded = self._encode(context) if context else [0]
                        x = torch.tensor([encoded], dtype=torch.long).to(self.device)
                        logits, _ = self.model(x)
                        probs = torch.softmax(logits, dim=-1)
                        _, top_indices = torch.topk(probs[0], k=min(10, self.vocab_size))

                        top_chars = []
                        for idx in top_indices.tolist():
                            char = self.idx_to_char.get(idx, ' ')
                            if char != self.pad_token and len(top_chars) < 3:
                                top_chars.append(char)

                        while len(top_chars) < 3:
                            for c in self.unigram_top:
                                if c not in top_chars:
                                    top_chars.append(c)
                                    break

                        preds.append(''.join(top_chars[:3]))
                    except Exception as e:
                        print(f'Warning: LSTM error for "{inp[:20]}...": {e}')
                        preds.append(''.join(self.unigram_top[:3]))
                else:
                    preds.append(''.join(self.unigram_top[:3]))

        print(f"N-gram hits: {ngram_hits}/{len(data)} ({100*ngram_hits/max(len(data),1):.1f}%)")
        return preds

    # ── Save / Load ───────────────────────────────────────────────────

    def save(self, work_dir):
        config = {
            'chars': self.chars,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'context_len': self.context_len,
            'is_trained': self.is_trained,
            'max_ngram_order': self.max_ngram_order,
            'unigram_top': self.unigram_top,
        }
        with open(os.path.join(work_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f)

        # Save n-gram model (convert int keys to str for JSON)
        ngram_data = {str(order): preds for order, preds in self.ngram_preds.items()}
        ngram_path = os.path.join(work_dir, 'ngram.json')
        with open(ngram_path, 'w', encoding='utf-8') as f:
            json.dump(ngram_data, f)
        size_mb = os.path.getsize(ngram_path) / (1024 * 1024)
        print(f'Saved n-gram model ({size_mb:.1f} MB)')

        if self.model is not None:
            model_path = os.path.join(work_dir, 'model.pt')
            torch.save(self.model.state_dict(), model_path)
            print(f'Saved LSTM model to {model_path}')

    @classmethod
    def load(cls, work_dir):
        model = cls()

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
            model.max_ngram_order = config.get('max_ngram_order', model.max_ngram_order)
            model.unigram_top = config.get('unigram_top', model.unigram_top)

        # Load n-gram model
        ngram_path = os.path.join(work_dir, 'ngram.json')
        if os.path.exists(ngram_path):
            with open(ngram_path, 'r', encoding='utf-8') as f:
                ngram_data = json.load(f)
            model.ngram_preds = {int(k): v for k, v in ngram_data.items()}
            total_entries = sum(len(v) for v in model.ngram_preds.values())
            print(f'Loaded n-gram model: {total_entries:,} entries across orders {sorted(model.ngram_preds.keys())}')
        else:
            print('No n-gram model found')

        # Load LSTM weights
        model_path = os.path.join(work_dir, 'model.pt')
        if os.path.exists(model_path):
            model._init_model()
            model.model.load_state_dict(torch.load(model_path, map_location=model.device, weights_only=True))
            model.model.eval()
            print(f'Loaded LSTM model from {model_path}')
        else:
            print('No LSTM model found, using n-gram only')

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
