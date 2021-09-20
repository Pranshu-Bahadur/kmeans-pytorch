import torch
from torch import tensor, multinomial, ones, stack, arange, where, nonzero, zeros, logical_xor, eq
from functools import singledispatch

"""
Types of seeders -> [+] uniform, [+]1_uniform+D^2, []oversampler(t?)
"""
# Pick k elements from given tensor according to uniform distribution.
def uniform_seeder(x : tensor, k : int, weights_func=None) -> tensor:
    seed = multinomial(ones(x.size(0))*(1./x.size(0)), k, replacement=False) \
            if weights_func is None else \
            multinomial(weights_func(x), k, replacement=False)
    indices = tensor(list(filter(lambda x: not any(x==seed), arange(0, x.size(0), 1).long().tolist())))
    hsplit = lambda x_, splits: [x_[split] for split in splits]
    return  hsplit(x, [seed, indices])

# Normal Kmeans cost
def varphi(x : tensor, c : tensor):
    distances = torch.pow(torch.cdist(x, c), 2)
    costs, indices = torch.min(distances, 1)
    return costs, indices

# Kmeans++ Seeder Method.
def uniform_d_squared_seeder(x : tensor, k : int) -> tensor:
    centers = []
    c, x = uniform_seeder(x, 1)
    centers += c
    while len(centers) < k:
        cost = sum(varphi(x, stack(centers))[0])
        c, x = uniform_seeder(x, 1, \
                lambda x_ : tensor(list(map(lambda p :\
                sum(varphi(tensor(p).unsqueeze(0), \
                stack(centers))[0])/cost, x_.tolist()))))
        centers += c
    return stack(centers), x
   


#def kmeans
