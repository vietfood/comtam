# Running a machine learning model on the GPU timeline

Dispatch model inference commands with a machine learning pass in a Metal 4 command buffer.

## Overview

This sample demonstrates how to encode commands that run a machine learning model with the Metal 4 API.
Each machine learning inference pass runs alongside other tasks on a Metal device and typically requires
less time and overhead than running a model from the CPU and passing its outputs to a Metal workload.

You encode a machine learning pass into a command buffer with the ``MTL4MachineLearningCommandEncoder`` protocol.
When the Metal device finishes running the pass, other passes can immediately work with the inference results.

The app multiplies two matrices on both the CPU and the Metal device, then reports whether the results are the same.
It does the matrix multiplication on the Metal device by running a machine learning pass with a model
that does matrix multiplication.
Metal apps can apply Core ML models in the form of a Metal package file,
Metal packages provide an entry point to the model within that a Metal device can run.

> Note: The sample's model is a relatively simple network that multiplies two matrices,
> a task which most apps can run more efficiently with the CPU or GPU shader code.

When you run the app, it:

1. Creates Metal resources, most of which are reusable
2. Compiles a machine learning-pipeline state from the model in the Metal package
3. Extracts tensor bindings from pipeline reflection
4. Creates tensors that match the binding requirements
5. Fills the input tensors with matrix data
6. Binds tensors to an argument table
7. Creates a heap for the machine learning encoder's temporary allocations
8. Encodes and runs the machine learning pass
9. Compares the GPU result against the CPU result

The app reports whether the results match and exits.

> Note: The app creates reusable resources once, such as the Metal device, compiler, command queue, and command buffer.

Long-running apps can follow the same pattern to avoid repeating setup costs.

For more information about the app and how it works, see
[Running a machine learning model on the GPU timeline](https://developer.apple.com/documentation/metal/running-a-machine learning-model-on-the-gpu-timeline)
in the developer documentation.
