import torch
import ninetoothed
import numpy as np
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)

@ninetoothed.jit
def add_kernel(
    x: Tensor(1).tile((BLOCK_SIZE,)),
    y: Tensor(1).tile((BLOCK_SIZE,)),
    z: Tensor(1).tile((BLOCK_SIZE,)),
):
    z = x + y

def add(x, y):
    lhs = torch.tensor(x, device='cuda')
    rhs = torch.tensor(y, device='cuda')

    output = torch.empty_like(lhs)
    add_kernel(lhs, rhs, output)
    return output

def main():
    size = 48
    x = np.random.rand(size) 
    y = np.random.rand(size) 
    print("the initial data is", x, y)
    output = add(x, y)

    print("the result is", output)

    lhs = torch.tensor(x, device='cuda')
    rhs = torch.tensor(y, device='cuda')
    assert torch.allclose(output, lhs + rhs)

if __name__ == "__main__":
    main()