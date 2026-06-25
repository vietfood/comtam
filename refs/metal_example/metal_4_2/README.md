# Running inline ML operations in a shader with Metal 4

Multiply matrices across multiple GPU cores with inline tensor operations.

This sample code project demonstrates how to run matrix multiplications in a GPU kernel function with inline tensor operations, which are available in Metal Shading Language 4 and later.
The sample app multiplies two matrices in a single GPU dispatch, and verifies the result by comparing it to a CPU-based reference implementation.

The sample's GPU kernel function runs the matrix multiplication in parallel by invoking a Metal 4 tensor operation, which distributes the workload across multiple GPU cores.
The kernel stores intermediate results in a cooperative tensor, which keeps the data in register memory, and avoids the latency of writing to device or threadgroup memory.
The sample's implementation of the kernel is minimal because Apple silicon GPUs don't require an explicit software pipeline to overlap memory and compute operations.
For GPUs in the [`apple10`](https://developer.apple.com/documentation/metal/mtlgpufamily/apple10) GPU family, and later, each core has a neural accelerator that improves runtime performance by running tensor operations directly in hardware.

> Note: For a list of chips in each GPU family, see [Metal Feature Set Tables (PDF)](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf).


For more information about the app and how it works, see
[Running inline ML operations in a shader with Metal 4](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4).
