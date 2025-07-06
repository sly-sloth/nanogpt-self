import torch
import os
import tiktoken
import math

from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from dataclasses import dataclass


dataset_dir = Path("dataset/harry-potter-books")

enc = tiktoken.get_encoding("gpt2")


@dataclass
class GPTConfig:
    block_size: int = 128
    n_embd: int = 64
    n_head: int = 4
    vocab_size: int = 50304
    n_layers: int = 4 # or change to 6 for betterment


class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.qkv_mat = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_mat(x)
        q, k, v = torch.split(qkv, self.config.n_embd, dim=-1)
        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)

        # print(f"inside MHA, {q.shape, k.shape, v.shape}, shapes of q, k, v")

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.shape[-1])
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, value=float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v

        # print(f"inside MHA, {y.shape}, shape of y")

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
    

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(2 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x
    

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha = MHA(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
    
    def forward(self, x):
        x = x + self.ln_1(self.mha(x))
        x = x + self.ln_2(self.mlp(x))
        return x
    

class hpGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            w_tok_emb = nn.Embedding(config.vocab_size, config.n_embd),
            w_pos_emb = nn.Embedding(config.block_size, config.n_embd),
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None): # idx : encoded text
        # print(f"inside forward func: {idx.shape}, shape of idx")
        # print("finding tok_emb")
        B, T = idx.shape
        tok_emb = self.transformer.w_tok_emb(idx) # B, T, C
        # print("finding pos_emb")
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.w_pos_emb(pos)

        # print(f"inside forward func : {tok_emb.shape, pos_emb.shape}, shapes of tok_emb and pos_emb")
        x = tok_emb + pos_emb

        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        
        return logits, loss
    
    def generate(self, start_string: str, num_return_sequences: int, max_length: int, device: torch.device):
        assert max_length <= self.config.block_size
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(start_string)
        tokens = torch.tensor(tokens)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)

        while xgen.shape[1] < max_length:
            with torch.no_grad():
                logits, _ = self(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                top50_values, top50_indices = torch.topk(probs, k=50, dim=-1)
                prob_indices = torch.multinomial(top50_values, num_samples=1)
                attached_tensor = torch.gather(top50_indices, dim=-1, index=prob_indices)
                xgen = torch.cat((xgen, attached_tensor), dim=1)
        
        for row_t in xgen:
            print(enc.decode(row_t.tolist()))


def load_tokens(filename):
    with open(filename, "r") as f:
        f_content = f.read()
    
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(f_content)
    tokens = torch.tensor(tokens, dtype=torch.long)
    return tokens


class HPDataloaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {"train", "val"}

        data_root = dataset_dir
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        
        self.reset()
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        if len(buf) != B*T + 1:
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
            buf = self.tokens[self.current_position : self.current_position + B*T + 1]
            
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # print(f"inside dataloader: {x.shape, y.shape}, shapes of x, y")


        self.current_position += B*T

        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0

        # if self.current_position == (self.current_parent_batch + 1) * 32 * B * T:
        #     if self.current_shard + 1 == len(self.shards):
        #         self.current_parent_batch = (self.current_parent_batch + 1) % 381
        #     self.current_shard = (self.current_shard + 1) % len(self.shards)
        #     self.tokens = load_tokens(self.shards[self.current_shard])
        #     self.current_position = self.current_parent_batch * 32 * B * T
            
        return x, y

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0
        self.current_parent_batch = 0

    def reset_from_config(self, current_shard, current_parent_batch):
        self.current_shard = current_shard
        self.current_parent_batch = current_parent_batch
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = current_parent_batch * 32 * self.B * self.T


max_lr = 3e-4
min_lr = 0.1 * max_lr
warmup_steps = 16
max_steps = 128

def get_lr(step):
    if step < warmup_steps:
        return min_lr + 0.9 * max_lr * step / warmup_steps
    else:
        coeff = (max_steps - step) / (max_steps - warmup_steps)
        coeff = math.sin(coeff * math.pi / 2)
        return min_lr + coeff * (max_lr - min_lr)
    

device = "cuda" if torch.cuda.is_available() else "cpu"

model = hpGPT(GPTConfig)
model.to(device)
# model = torch.compile(model)


batch_size = 4096 * 2
B = 8 # can increase in case of gpu training
T = GPTConfig.block_size
grad_accum_steps = batch_size //(B*T)

train_loader = HPDataloaderLite(B, T, "train")
val_loader = HPDataloaderLite(B, T, "val")

optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-4)

iters = []
loss_list = []

for step in range(max_steps):
    val_loss = 0.0
    if step % 8 == 0:
        model.eval()
        with torch.no_grad():
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # print(f"inside train loop: {x.shape, y.shape}, shapes of x, y")

            logits, val_loss = model(x, y)
        print(f"val loss: {val_loss.item():.4f}")
        print()

    model.train()
    loss_accum = 0.0
    optimizer.zero_grad()
    for grad_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        logits, loss = model(x, y)
        loss /= grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

        # print("grad accum step over")
        # print()
    
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    optimizer.step()
    iters.append(step)
    loss_list.append(loss_accum.cpu().item())

    if step % 4 == 0 or step == max_steps-1:
        print(f"step: {step}, avg_loss: {loss_accum.item():.4f}, lr: {lr:.4e}")
    
    if step % 16 == 0 or step == max_steps-1:
        model.eval()
        print("generating text...")
        model.generate("Your house is Slytherin!", 3, 50, device)
        print()
        print()
    
    # print()
    # print()
        