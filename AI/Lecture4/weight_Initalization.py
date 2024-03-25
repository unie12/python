import numpy as np
from numpy.random import randn

dims = [4096] * 7
hs = []
x = np.random.randn(16, dims[0])

# Activation statisctics
# -> no learning
# for Din, Dout in zip(dims[:-1], dims[1:]):
#     w = 0.01 * np.random.randn(Din, Dout)
#     w = 0.05 * np.random.randn(Din, Dout)
#     x = np.tanh(x.dot(w))
#     hs.append(x)

# Xavier Initialziation
# -> activations are nicely scaled for all layers
# for Din, Dout in zip(dims[:-1], dims[1:]):
#     w = np.random.randn(Din, Dout) / np.sqrt(Din)
#     x = np.maximum(x.dot(w))
#     hs.append(x)

    
# ReLU
# no learning
# for Din, Dout in zip(dims[:-1], dims[1:]):
#     w = np.random.randn(Din, Dout) / np.sqrt(Din)
#     x = np.maximum(0,x.dot(w))
#     hs.append(x)

# Kaiming / MSRA initialization
# Activations are nicley scaled for all layers
for Din, Dout in zip(dims[:-1], dims[1:]):
    w = np.random.randn(Din, Dout) * np.sqrt(2/Din)
    x = np.maximum(0,x.dot(w))
    hs.append(x)