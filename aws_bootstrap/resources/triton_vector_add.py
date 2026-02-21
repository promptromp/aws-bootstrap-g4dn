"""Example Python file using triton to implement a custom vector addition kernel.

Mainly used as validation that our EC2 instance has proper Python venv and pytorch correctly set up,
and that we can run triton kernels on the GPU.

Reference: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html

"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE

    # array of pointers to elements we'll process in this program instance.
    offsets = start + tl.arange(0, BLOCK_SIZE)

    # mask out out-of-bounds elements.
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    out = x + y

    # write back to DRAM / HBM
    tl.store(out_ptr + offsets, out, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


def main():
    torch.manual_seed(42)

    device = triton.runtime.driver.active.get_active_torch_device()
    print(f"{device=}")

    size = 2**20

    # note: the input tensors must be on the same device as the triton kernel
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    output_torch = x + y
    output_triton = add(x, y)

    print(f"{output_torch=}")
    print(f"{output_triton=}")
    assert torch.allclose(output_torch, output_triton, atol=1e-6), "The outputs from torch and triton don't match!"
    print("Success!")


if __name__ == "__main__":
    main()
