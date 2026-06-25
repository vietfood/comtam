/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Utility methods for working with Metal tensor extents.
*/

#import "Matrix+TensorUtilities.h"

@implementation Matrix (TensorUtilities)

/// Calculates the total number of elements in a tensor.
///
/// Call this method to calculate the product of all dimension extents for a tensor.
/// For example, a tensor with dimensions `[4, 3]` has 12 total elements (4 x 3).
///
/// This is a convenience method for when you need to allocate memory or iterate
/// over all elements in a tensor.
///
/// - Parameter dimensions: The dimensions of a tensor.
/// - Returns: The total number of elements in the tensor.
+ (NSInteger)totalElementsForDimensions:(MTLTensorExtents *)dimensions {
    NSInteger totalElements = 1;
    for (NSUInteger index = 0; index < dimensions.rank; index++) {
        totalElements *= [dimensions extentAtDimensionIndex:index];
    }
    return totalElements;
}


/// Creates a zero-valued tensor extents instance for any tensor rank.
///
/// Use this method to create the origin point when copying data into a tensor slice.
/// The returned extents has all dimensions set to zero, representing the starting
/// position of a tensor slice (For example, `[0, 0]` for rank 2, `[0, 0, 0]` for rank 3).
///
/// This is a convenience method for the common pattern of providing the origin
/// when calling methods like `replaceSliceOrigin:sliceDimensions:withBytes:strides:`.
///
/// - Parameter rank: The number of dimensions for the tensor extents.
/// - Returns: A tensor extents instance with all values set to zero.
+ (MTLTensorExtents *)tensorSliceOriginForRank:(NSUInteger)rank {
    NSInteger zeroValues[MTL_TENSOR_MAX_RANK] = { 0 };
    return [[MTLTensorExtents alloc] initWithRank:rank values:zeroValues];
}

/// Creates a stride tensor extents instance for a contiguous, row-major memory layout.
///
/// Call this method to configure default strides for copying data between matrices and tensors.
/// The method calculates strides starting with the innermost dimension (index 0) equal to `1`,
/// and each subsequent dimension's stride is the product of all previous dimension extents.
/// For example, a tensor with dimensions `[4, 3]` has default strides `[1, 4]`.
///
/// This is a convenience method for when you need to provide strides for
/// methods like `replaceSliceOrigin:sliceDimensions:withBytes:strides:`
/// or `getBytes:strides:fromSliceOrigin:sliceDimensions:`.
///
/// - Parameter dimensions: The dimensions of a tensor.
/// - Returns: A tensor extents instance that has the default strides for a tensor
/// with `dimensions`.
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
