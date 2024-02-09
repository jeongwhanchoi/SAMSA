import torch
import torch.nn as nn

@torch.no_grad()
def conditional_gumbel(logits, D, k=1):
    """Outputs k samples of Q = StandardGumbel(), such that argmax(logits
    + Q) is given by D (one hot vector)."""
    # iid. exponential
    E = torch.distributions.exponential.Exponential(rate=torch.ones_like(logits)).sample([k])
    # E of the chosen class
    Ei = (D * E).sum(dim=-1, keepdim=True)
    # partition function (normalization constant)
    Z = logits.exp().sum(dim=-1, keepdim=True)
    # Sampled gumbel-adjusted logits
    adjusted = (D * (-torch.log(Ei) + torch.log(Z)) +
                (1 - D) * -torch.log(E/torch.exp(logits) + Ei / Z))
    return adjusted - logits


def exact_conditional_gumbel(logits, D, k=1):
    """Same as conditional_gumbel but uses rejection sampling."""
    # Rejection sampling.
    idx = D.argmax(dim=-1)
    gumbels = []
    while len(gumbels) < k:
        gumbel = torch.rand_like(logits).log().neg().log().neg()
        if logits.add(gumbel).argmax() == idx:
            gumbels.append(gumbel)
    return torch.stack(gumbels)


def replace_gradient(value, surrogate):
    """Returns `value` but backpropagates gradients through `surrogate`."""
    return surrogate + (value - surrogate).detach()


def gumbel_rao(logits, k, temp=1.0, I=None):
    """Returns a categorical sample from logits (over axis=-1) as a
    one-hot vector, with gumbel-rao gradient.

    k: integer number of samples to use in the rao-blackwellization.
    1 sample reduces to straight-through gumbel-softmax.

    I: optional, categorical sample to use instead of drawing a new
    sample. Should be a tensor(shape=logits.shape[:-1], dtype=int64).

    """
    num_classes = logits.shape[-1]
    if I is None:
        I = torch.distributions.categorical.Categorical(logits=logits).sample()
    D = torch.nn.functional.one_hot(I, num_classes).float()
    adjusted = logits + conditional_gumbel(logits, D, k=k)
    surrogate = torch.nn.functional.softmax(adjusted/temp, dim=-1).mean(dim=0)
    return replace_gradient(D, surrogate)

class GRSoftmax(nn.Module):
    def __init__(self, temp=0.1, k=10):
        super(GRSoftmax, self).__init__()
        self.k = k
        self.temp = temp

    def forward(self, x, dim):
        x = torch.transpose(x, dim, len(x.shape) - 1)
        x = gumbel_rao(x, self.k, self.temp)
        x = torch.transpose(x, dim, len(x.shape) - 1)
        return x