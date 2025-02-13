import torch
import numpy as np

from hashlib import sha256

def default_hash_fn(tensor):
    return int(sha256(str(tensor).encode('utf-8')).hexdigest(), 16) % (10 ** 8)


@torch.no_grad()
def generate(model, prior_tokens, max_length=200, watermark=True, gamma=0.5, delta=2, hash_function=default_hash_fn):
    B, T = prior_tokens.shape
    device = prior_tokens.device

    generated_tokens = prior_tokens
    for _ in range(max_length - T):
        l_t = model(generated_tokens)[:, -1, :]

        if watermark:
            seeds = [hash_function(generated_tokens[i, -1]) for i in range(B)]
            generators = [torch.Generator(device=device).manual_seed(seed)
                          for seed in seeds]

            vs = l_t.shape[-1] 
            gls = int(gamma * vs)  
            gli = torch.stack([torch.randperm(vs, generator=generators[i], device=device)
                               for i in range(B)]) 
            l_t = l_t + delta * (gli < gls)

        l_t = torch.softmax(l_t, dim=-1)
        next_tokens = torch.multinomial(l_t, 1)
        generated_tokens = torch.cat([generated_tokens, next_tokens], dim=-1)

    return generated_tokens


def detect_watermark(ids, vocab_size, gamma=0.5, hash_function=default_hash_fn):
    B, T = ids.shape
    device = ids.device
    gls = int(gamma * vocab_size)  
    in_green_list = torch.zeros(B, dtype=torch.float32).to(device) 

    for i in range(T-1):
        seeds = [hash_function(ids[j, i]) for j in range(B)]
        generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]

        gli = torch.stack([torch.randperm(vocab_size, generator=generators[i], device=device)
                           for i in range(B)]) 

        in_green_list += (gli.gather(1, ids[:, i+1].unsqueeze(-1)) < gls).squeeze()
        
    z = (in_green_list - gamma * T) / np.sqrt(T*gamma*(1-gamma))
    return z

@torch.no_grad()
def get_perplexities(model, ids):
    B, T = ids.shape
    
    perplexities = torch.zeros(B).to(ids.device)
    for i in range(T-1):
        l_t = model(ids[:, :i+1])[:, -1, :]
        l_t = torch.softmax(l_t, dim=-1)
        l_t = l_t[range(B), ids[:, i+1]]
        l_t = torch.log(l_t)
        perplexities += l_t
    
    return torch.exp(-perplexities / (T-1))
