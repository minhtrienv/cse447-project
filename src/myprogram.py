#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import torch.nn as nn


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    TRAIN_TXT_DEFAULT = os.environ.get("TRAIN_TXT", os.path.join("example", "train.txt"))
    CKPT_NAME = "model.checkpoint"   # keep same filename

    @staticmethod
    def _device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _build_vocab(text: str):
        # char-level vocab: just all chars seen in train
        # reserve 2 specials
        specials = ["<pad>", "<unk>"]
        chars = sorted(set(text))
        itos = specials + chars
        stoi = {ch: i for i, ch in enumerate(itos)}
        return stoi, itos

    @staticmethod
    def _encode(text: str, stoi):
        unk = stoi["<unk>"]
        return torch.tensor([stoi.get(ch, unk) for ch in text], dtype=torch.long)
    class _CharLSTM(nn.Module):
        def __init__(self, vocab_size, emb_dim=64, hidden_dim=256, num_layers=2):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim)
            self.lstm = nn.LSTM(
                input_size=emb_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,   # (B, T, E)
            )
            self.proj = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x, state=None):
            # x: (B, T) -> logits: (B, T, V)
            e = self.emb(x)
            y, state = self.lstm(e, state)
            logits = self.proj(y)
            return logits, state

    def __init__(self):
        self.device = self._device()
        self.stoi = None
        self.itos = None
        self.net = None
        self.cfg = {
            "emb_dim": 64,
            "hidden_dim": 256,
            "num_layers": 2,
            "seq_len": 128,
            "batch_size": 64,
            "epochs": 2,
            "lr": 2e-3,
            "grad_clip": 1.0,
            "max_context": 512,
            "pred_len": 3,   # predict next 3 chars, matches original output shape idea
        }

    @classmethod
    def load_training_data(cls):
        # your code here
        # NEW: read plain text (e.g., OSCAR already extracted to text)
        train_path = cls.TRAIN_TXT_DEFAULT
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"Training file not found: {train_path}. "
                f"Put a plain text file there (example/train.txt) or set TRAIN_TXT=/path/to/train.txt"
            )
        with open(train_path, "rt", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return text

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        # data is a big training string
        torch.manual_seed(0)

        # build vocab + model
        self.stoi, self.itos = self._build_vocab(data)
        vocab_size = len(self.itos)
        self.net = self._CharLSTM(
            vocab_size,
            emb_dim=self.cfg["emb_dim"],
            hidden_dim=self.cfg["hidden_dim"],
            num_layers=self.cfg["num_layers"],
        ).to(self.device)

        ids = self._encode(data, self.stoi)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg["lr"])
        loss_fn = nn.CrossEntropyLoss()

        def sample_batch(ids_1d):
            N = ids_1d.numel()
            T = self.cfg["seq_len"]
            B = self.cfg["batch_size"]
            max_start = N - (T + 1)
            if max_start <= 0:
                raise ValueError("Training text too small for seq_len.")
            starts = torch.randint(0, max_start, (B,))
            x = torch.stack([ids_1d[s:s+T] for s in starts], dim=0)        # (B,T)
            y = torch.stack([ids_1d[s+1:s+1+T] for s in starts], dim=0)    # (B,T)
            return x.to(self.device), y.to(self.device)

        self.net.train()
        steps_per_epoch = max(1, ids.numel() // (self.cfg["seq_len"] * self.cfg["batch_size"]))
        for ep in range(self.cfg["epochs"]):
            for step in range(steps_per_epoch):
                x, y = sample_batch(ids)
                logits, _ = self.net(x)  # (B,T,V)
                loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg["grad_clip"] is not None:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg["grad_clip"])
                opt.step()

                if (step + 1) % 200 == 0:
                    print(f"[train] ep={ep+1} step={step+1}/{steps_per_epoch} loss={loss.item():.4f}")

    def run_pred(self, data):
        # your code here
        # Predict next 3 chars for each line (greedy)
        assert self.net is not None and self.stoi is not None and self.itos is not None, "Model not loaded."
        self.net.eval()

        preds = []
        unk = self.stoi["<unk>"]
        V = len(self.itos)

        with torch.no_grad():
            for inp in data:
                if len(inp) == 0:
                    inp = " "
                context = inp[-self.cfg["max_context"]:]
                x = torch.tensor([[self.stoi.get(ch, unk) for ch in context]],
                                 dtype=torch.long, device=self.device)

                logits, state = self.net(x)              # (1,T,V)
                last = logits[:, -1, :]                  # (1,V)

                out = []
                for _ in range(self.cfg["pred_len"]):
                    nxt = int(torch.argmax(last, dim=-1).item())
                    out.append(self.itos[nxt])

                    step_x = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
                    step_logits, state = self.net(step_x, state)   # feed back in
                    last = step_logits[:, -1, :]

                preds.append("".join(out))

        self.net.train()
        return preds

    def save(self, work_dir):
        # your code here
        # save torch checkpoint + vocab
        os.makedirs(work_dir, exist_ok=True)
        ckpt_path = os.path.join(work_dir, self.CKPT_NAME)
        payload = {
            "cfg": self.cfg,
            "stoi": self.stoi,
            "itos": self.itos,
            "state_dict": None if self.net is None else self.net.state_dict(),
        }
        torch.save(payload, ckpt_path)

    @classmethod
    def load(cls, work_dir):
        # your code here
        ckpt_path = os.path.join(work_dir, cls.CKPT_NAME)
        payload = torch.load(ckpt_path, map_location="cpu")

        m = cls()
        m.cfg = payload["cfg"]
        m.stoi = payload["stoi"]
        m.itos = payload["itos"]

        vocab_size = len(m.itos)
        m.net = m._CharLSTM(
            vocab_size,
            emb_dim=m.cfg["emb_dim"],
            hidden_dim=m.cfg["hidden_dim"],
            num_layers=m.cfg["num_layers"],
        ).to(m.device)
        m.net.load_state_dict(payload["state_dict"])
        m.net.eval()
        return m


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
