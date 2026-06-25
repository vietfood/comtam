/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
A matrix class that encapsulates matrix data and operations.
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

/// A minimal matrix class that works with 2D tensors.
///
/// Create a matrix with random data by calling the ``matrixWithRandomRows:columns:`` method.
@interface Matrix : NSObject

/// The number of rows in the matrix.
@property (nonatomic, readonly) NSInteger rows;

/// The number of columns in the matrix.
@property (nonatomic, readonly) NSInteger columns;

/// Creates a matrix with random floating-point values.
/// - Parameters:
///   - rows: The number of rows in the new matrix.
///   - columns: The number of columns in the new matrix.
+ (instancetype)matrixWithRandomRows:(NSInteger)rows
                             columns:(NSInteger)columns;

/// Creates a matrix with every element equal to zero.
/// - Parameters:
///   - rows: The number of rows in the new matrix.
///   - columns: The number of columns in the new matrix.
- (instancetype)initWithRows:(NSInteger)rows
                     columns:(NSInteger)columns;

/// Creates a matrix from a Metal tensor.
/// - Parameter tensor: A tensor that has a rank of `2` and a data type equal to
/// `MTLTensorDataTypeFloat32`.
/// - Returns: A matrix if the method succeeds; otherwise, `nil`.
- (nullable instancetype)initFromTensor:(id<MTLTensor>)tensor;

/// Multiplies this matrix with another matrix.
/// - Parameter other: A matrix the method multiplies with this matrix.
/// - Returns: A product matrix if the method succeeds; otherwise, `nil`.
- (nullable Matrix *)multiplyWithMatrix:(Matrix *)other;

/// Compares this matrix with another matrix.
///
/// The method applies a tolerance of `1e-5`, which means it considers two values
/// equal if their difference's magnitude is less than or equal to the tolerance value.
/// - Parameter other: A matrix the method compares with this matrix.
/// - Returns: A Boolean value that indicates whether the matrices are equal within a small tolerance.
- (BOOL)isEqualToMatrix:(Matrix *)other;

/// Prints the matrix to the console.
/// - Parameter name: A name for the matrix the method prints above its data.
- (void)printWithName:(NSString *)name;

/// Copies this matrix's data into a Metal tensor.
/// - Parameter tensor: A tensor that receives the matrix data.
- (void)copyDataToTensor:(id<MTLTensor>)tensor;

@end

NS_ASSUME_NONNULL_END
