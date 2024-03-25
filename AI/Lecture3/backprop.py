# Modularized API
class ComputationalGraph(object):
    def forward(inputs):
        # 1. pass inputs to input gates
        # 2. forward the computaional graph
        for gate in self.graph.nodes_topologically_sorted():
            gate.forward()
        return loss
    def backward():
        for gate in reversed(self.graph.nodes_topologically_sorted()):
            gate.backward()
        return inputs_gradients
    
# forward / backward API
class Multiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        z = x * y
        return z
    @staticmethod
    def backward(ctx, grad_z):
        x, y = ctx.saved_tensors
        # multiply upstream and local gradients
        grad_x = y * grad_z
        grad_y = x  grad_z
        return grad_x, grad_y