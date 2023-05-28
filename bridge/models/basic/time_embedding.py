import torch
import torch.nn.functional as F
import math


def get_timestep_embedding(timesteps, embedding_dim=128, max_period=10000):
    """
      From Fairseq.
      Build sinusoidal embeddings.
      This matches the implementation in tensor2tensor, but differs slightly
      from the description in Section 3.5 of "Attention Is All You Need".
      https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py
    """
    half_dim = embedding_dim // 2
    emb = math.log(max_period) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)

    emb = timesteps * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, [0,1])

    return emb


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    test = get_timestep_embedding(torch.linspace(0, 0.1, 10).reshape(-1, 1))
    plt.subplot(1, 2, 1)
    plt.plot(test.T[:test.shape[1]//2])
    plt.subplot(1, 2, 2)
    plt.plot(test.T[test.shape[1]//2:])
    plt.show()