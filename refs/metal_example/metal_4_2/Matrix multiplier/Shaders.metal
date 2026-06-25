/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
The app's source code for its Metal shaders.
*/

#include <metal_stdlib>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

/// Clamps negative values to zero.
///
/// - Parameter value: A half-precision value.
///
/// - Returns: The value if positive; otherwise, `0.0`.
///
/// The rectified linear unit (ReLU) activation function returns a positive value
/// as-is without modification, but returns `0` for a negative value.
half relu(half value) {
    return max(value, (half)0.0f);
}

/// Multiplies two input matrices and applies a ReLU activation function to the product.
///
/// - Parameters:
///   - threadgroupIdentifier: The position of this threadgroup in the compute grid.
///   - sourceA: A tensor that contains the first input matrix.
///   - sourceB: A tensor that contains the second input matrix.
///   - destination: A tensor that receives the product matrix.
///
/// Each threadgroup computes a 64 x 64 tile of the output by slicing each tensor to the
/// portion that threadgroup is responsible for.
/// The input slices use `dynamic_extent` for the inner dimension so the compiler infers
/// it at runtime, and static extents for the tile dimensions to skip per-element bounds checks.
///
/// The kernel captures the matrix multiplication output in a cooperative tensor,
/// which distributes the output elements across the threads in the threadgroup
/// and stores them in register memory, which avoids the latency of writing to device or threadgroup memory.
///
/// Applying the ReLU activation function in the same dispatch avoids the cost of a
/// separate GPU dispatch.
kernel void
matrix_multiplication_kernel(uint2 threadgroupIdentifier [[threadgroup_position_in_grid]],
                             tensor<device half, dextents<int, 2>> sourceA,
                             tensor<device half, dextents<int, 2>> sourceB,
                             tensor<device half, dextents<int, 2>> destination)
{
    // Slice each tensor to the tile this threadgroup computes.
    const int TileSize = 64;
    int tileOriginX = TileSize * threadgroupIdentifier.x;
    int tileOriginY = TileSize * threadgroupIdentifier.y;

    auto sliceA = sourceA.slice<dynamic_extent, TileSize>(0, tileOriginY);
    auto sliceB = sourceB.slice<TileSize, dynamic_extent>(tileOriginX, 0);
    auto destinationSlice = destination.slice<TileSize, TileSize>(tileOriginX, tileOriginY);

    // Describe the tile size and inner dimension for the matrix multiplication.
    constexpr auto descriptor = matmul2d_descriptor(TileSize,
                                                    TileSize,
                                                    dynamic_length_v<int>);

    // Create a 2D matrix multiplication operation with four SIMD groups.
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    // Capture the matrix multiplication output in a cooperative tensor.
    auto cooperativeTensor = operation.get_destination_cooperative_tensor<decltype(sliceA),
                                                                          decltype(sliceB),
                                                                          half>();

    // Run the matrix multiplication.
    operation.run(sliceA, sliceB, cooperativeTensor);
    
    // Apply the ReLU activation function to each element.
    auto threadElements = cooperativeTensor.get_capacity();
    for (int element = 0; element < threadElements; element++)
    {
        auto value = cooperativeTensor[element];
        cooperativeTensor[element] = relu(value);
    }

    // Store the results to the destination slice.
    cooperativeTensor.store(destinationSlice);
}
