/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
A matrix class that encapsulates matrix data and operations.
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

/// A minimal matrix class that works with 2D tensors using 16-bit floating-point precision.
///
/// Create a matrix with random data by calling the ``matrixWithRandomRows:columns:`` method.
@interface Matrix : NSObject

/// The number of rows in the matrix.
@property (nonatomic, readonly) NSInteger rows;

/// The number of columns in the matrix.
@property (nonatomic, readonly) NSInteger columns;

/// Creates a matrix with random floating-point values.
///
/// - Parameters:
///   - rows: The number of rows in the new matrix.
///   - columns: The number of columns in the new matrix.
///
/// Each element in the matrix is in the range `[-1.0, 1.0]`.
+ (instancetype)matrixWithRandomRows:(NSInteger)rows
                             columns:(NSInteger)columns;

/// Creates a matrix with every element equal to zero.
///
/// - Parameters:
///   - rows: The number of rows in the new matrix.
///   - columns: The number of columns in the new matrix.
- (instancetype)initWithRows:(NSInteger)rows
                     columns:(NSInteger)columns;

/// Creates a matrix from a Metal tensor.
///
/// - Parameter tensor: A tensor with a rank of `2` and a data type of `MTLTensorDataTypeFloat16`.
///
/// - Returns: A matrix if the method succeeds; otherwise, `nil`.
- (nullable instancetype)initFromTensor:(id<MTLTensor>)tensor;

/// Multiplies this matrix with another matrix.
/// 
/// - Parameter multiplier: Another matrix the method multiplies with this matrix, the multiplicand.
///
/// - Returns: A product matrix if the method succeeds; otherwise, `nil`.
///
/// The method clears all negative values in the product matrix's elements to zero
/// to replicate the kernel function's rectified linear unit (ReLU) activation function.
- (nullable Matrix *)multiplyWithMatrix:(Matrix *)other;

/// Compares this matrix with another matrix.
///
/// - Parameter other: A matrix the method compares with this matrix.
///
/// - Returns: A Boolean value that indicates whether the matrices are equal within tolerance.
///
/// The method uses a tolerance of `0.05` for 16-bit floating-point comparisons.
- (BOOL)isEqualToMatrix:(Matrix *)other;

/// Prints a preview of the matrix.
///
/// - Parameter name: A name for the matrix, which the method prints above its data.
///
/// The method only prints up to the first four rows and columns.
- (void)printWithName:(NSString *)name;

/// Copies this matrix's data into a Metal tensor.
///
/// - Parameter tensor: A tensor that receives the matrix data.
- (void)copyDataToTensor:(id<MTLTensor>)tensor;

@end

NS_ASSUME_NONNULL_END
