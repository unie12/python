import torch

EPOCH = 5
PATH = "model.pt"
LOSS = 0.4

torch.save({
    'epoch': EPOCH,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': LOSS
}, PATH)