import torch
import numpy as np

p = 0.5

def train_step(X):

    # forward pass for example 3-layer neural network
    H1 = np.maximum(0, np.dot(W1, X) + b1)
    U1 = np.random.rand(*H1.shape) < p # first dropout mask
    H1 *= U1 # drop
    H2 = np.maximum(0, np.dot(W2, H1) + b2)
    U2 = np.random.rand(*H2.shape) < p # second dropout mask
    H2 *= U2 # drop
    out = np.dot(W3, H2) + b3


# dropout: test time
# output at test time = expected output at training time
def predict(X):
    # ensembled forward pass
    H1 = np.maximum(0, np.dot(W1, X) + b1) * p # scale the activations
    H1 = np.maximum(0, np.dot(W2, H1) + b2) * p # scale the activations
    out = np.dot(W3, H2) + b3