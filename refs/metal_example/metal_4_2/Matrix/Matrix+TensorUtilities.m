/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Utility methods for working with Metal tensor extents.
*/

#import "Matrix+TensorUtilities.h"

@implementation Matrix (TensorUtilities)

/// Calculates the total number of elements in a tensor.
///
/// - Parameter dimensions: The dimensions of a tensor.
///
/// - Returns: The total number of elements in the tensor.
///
/// For example, a tensor with dimensions `[4, 3]` has 12 total elements.
+ (NSInteger)totalElementsForDimensions:(MTLTensorExtents *)dimensions {
    NSInteger totalElements = 1;
    for (NSUInteger index = 0; index < dimensions.rank; index++) {
        totalElements *= [dimensions extentAtDimensionIndex:index];
    }
    return totalElements;
}

/// Creates a zero-valued tensor extents instance for any tensor rank.
///
/// - Parameter rank: The number of dimensions for the tensor extents.
///
/// - Returns: A tensor extents instance with all values set to zero.
///
/// The returned extents has all dimensions set to zero, representing the starting
/// position of a tensor slice --- for example, `[0, 0]` for rank 2 or `[0, 0, 0]` for rank 3.
+ (MTLTensorExtents *)tensorSliceOriginForRank:(NSUInteger)rank {
    NSInteger zeroValues[MTL_TENSOR_MAX_RANK] = { 0 };
    return [[MTLTensorExtents alloc] initWithRank:rank values:zeroValues];
}

/// Creates a stride tensor extents instance for a contiguous, row-major memory layout.
///
/// - Parameter dimensions: The dimensions of a tensor.
///
/// - Returns: A tensor extents instance that has the default strides for a tensor
/// with `dimensions`.
///
/// The innermost dimension (index 0) has a stride of `1`, and each subsequent
/// dimension's stride is the product of all previous dimension extents.
/// For example, a tensor with dimensions `[4, 3]` has default strides `[1, 4]`.
+ (MTLTensorExtents *)tensorStridesForDimensions:(MTLTensorExtents *)dimensions {
    NSInteger strideValues[MTL_TENSOR_MAX_RANK];
    NSInteger stride = 1;
    for (NSInteger index = 0; index < dimensions.rank; index++) {
        strideValues[index] = stride;
        stride *= [dimensions extentAtDimensionIndex:index];
    }
    return [[MTLTensorExtents alloc] initWithRank:dimensions.rank
                                           values:strideValues];
}

@end
