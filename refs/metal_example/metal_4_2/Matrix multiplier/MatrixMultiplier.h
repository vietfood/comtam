/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
A class that multiplies matrices with a cooperative tensor in shader code.
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "Matrix.h"

NS_ASSUME_NONNULL_BEGIN

/// A class that manages Metal objects and multiplies matrices using cooperative tensors.
///
/// Create a matrix multiplier, configure it with ``configureWithMatrix1:matrix2:``,
/// and run the computation with ``encodeAndRunComputation``.
/// Retrieve the product tensor with the ``productTensor`` method.
@interface MatrixMultiplier : NSObject

/// Creates a matrix multiplier.
- (instancetype)init;

/// Prepares the multiplier to run a matrix multiplication with cooperative tensors
/// on a Metal device.
/// - Parameters:
///   - matrix1: A matrix for the first input.
///   - matrix2: A matrix for the second input.
/// - Returns: A Boolean value that indicates whether the method succeeds.
- (BOOL)configureWithMatrix1:(Matrix *)matrix1
                     matrix2:(Matrix *)matrix2;

/// Encodes a compute pass and runs it on the GPU.
- (void)encodeAndRunComputation;

/// Returns the product tensor from the GPU computation.
- (id<MTLTensor>)productTensor;

@end

NS_ASSUME_NONNULL_END
