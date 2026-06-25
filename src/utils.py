import torch as t
from torch.utils.data import Dataset, DataLoader, random_split

from dataclasses import dataclass

import numpy as np

from transformer_lens import HookedTransformer, HookedTransformerConfig


vocab = {str(i): i for i in range(10)}
vocab['='] = 10
vocab['+'] = 11

vocab_inv = {i : str(i) for i in range(10)}
vocab_inv[10] = '='
vocab_inv[11] = '+'


def pad(num: int, reversed: bool=False) -> str:
    num_str = str(num)
    while len(num_str) < 4:
        num_str = '0' + num_str

    if reversed:
            num_str = num_str[::-1]

    return num_str

def generate_sample(cfg) -> list:
    max = cfg.max_output

    d1 = np.random.randint(cfg.min_output, cfg.max_output)
    d2 = np.random.randint(cfg.min_output, cfg.max_output)

    sum = d1 + d2

    return f'{pad(d1)}+{pad(d2)}={pad(sum, reversed=True)}'


@dataclass
class dataConfig:
    min_output: int = 0
    n_digits_input = 3
    n_digits_output = 4
    max_output: int = 5000
    n_samples: int = 50000
data_cfg = dataConfig()


# Generate samples, tokenize, and stack on top of a new batch dimension.

samples_tensor = t.stack(
    [t.tensor([vocab[char] for char in generate_sample(data_cfg)]) for n in range(0, data_cfg.n_samples)]
    ,dim=0)


class SumDataset(Dataset):
    def __init__(self : Dataset, token_dict : dict, samples, carry=True):
        self.token_dict = token_dict
        self.samples = samples
        self.carry = carry

    def __getitem__(self, idx):
        return {"tokens": self.samples[idx, :]}

    def __len__(self):
        return self.samples.shape[0]


ds = SumDataset(vocab, samples_tensor)
cutoffs = [0.8, 0.2]
generator = t.Generator().manual_seed(42)

train_ds, val_ds = random_split(ds, cutoffs, generator=generator)

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=128, shuffle=True, drop_last=True)


def create_model(
    num_digits: int,
    seed: int,
    d_model: int,
    d_head: int,
    n_layers: int,
    n_heads: int,
    d_mlp: int | None,
    normalization_type: str | None,
    device: str = "cuda",
    **kwargs,  # ignore other kwargs
) -> HookedTransformer:
    t.manual_seed(seed)
    np.random.seed(seed)

    attn_only = d_mlp is None

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_ctx=num_digits * 3 + 2,
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        d_mlp=d_mlp,
        attn_only=attn_only,
        act_fn="relu",
        d_vocab=12,
        use_attn_result=True,
        use_split_qkv_input=True,
        use_hook_tokens=True,

        normalization_type=normalization_type,
        device=device,
    )

    model = HookedTransformer(cfg)
    return model
