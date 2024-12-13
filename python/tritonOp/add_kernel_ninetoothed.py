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

def add(x: list, y: list) -> torch.Tensor:
    print("add func start")
    lhs = torch.tensor(x, device='cuda')
    rhs = torch.tensor(y, device='cuda')

    output = torch.empty_like(lhs)
    add_kernel(lhs, rhs, output)
    return output

def main():
    size = 48
    x = np.random.rand(size) 
    y = np.random.rand(size) 
    # x = [1, 2, 3]
    # y = [4, 5, 6]
    print("the initial data is", x, y)
    output = add(x, y)

    print("the result is", output)

    lhs = torch.tensor(x, device='cuda')
    rhs = torch.tensor(y, device='cuda')
    assert torch.allclose(output, lhs + rhs)

if __name__ == "__main__":
    main()