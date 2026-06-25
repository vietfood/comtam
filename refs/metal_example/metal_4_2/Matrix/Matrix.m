/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Implements a matrix class that multiplies matrices on the CPU with ReLU activation.
*/

#import "Matrix.h"
#import "Matrix+TensorUtilities.h"

typedef __fp16 half;

@interface Matrix () {
    /// The matrix data as a contiguous array of 16-bit floats.
    NSMutableData *_data;
}
@end

@implementation Matrix

- (NSData *)data {
    return _data;
}

// MARK: - Factory methods

/// Creates a matrix with random floating-point values.
///
/// - Parameters:
///   - rows: The number of rows in the new matrix.
///   - columns: The number of columns in the new matrix.
///
/// Each element in the matrix is in the range `[-1.0, 1.0]`.
+ (instancetype)matrixWithRandomRows:(NSInteger)rows
                             columns:(NSInteger)columns {
    Matrix *matrix = [[self alloc] initWithRows:rows columns:columns];
    [matrix fillWithRandomData];
    return matrix;
}

// MARK: - Initializers

/// Creates a matrix with every element equal to zero.
///
/// - Parameters:
///   - rows: The number of rows in the new matrix.
///   - columns: The number of columns in the new matrix.
- (instancetype)initWithRows:(NSInteger)rows columns:(NSInteger)columns {
    self = [super init];
    if (self == nil) {
        return self;
    }

    _rows = rows;
    _columns = columns;
    _data = [NSMutableData dataWithLength:(rows * columns) * sizeof(half)];

    return self;
}

/// Creates a matrix with existing data.
///
/// - Parameters:
///   - data: The matrix data as a contiguous array of 16-bit floats.
///   - rows: The number of rows in the matrix.
///   - columns: The number of columns in the matrix.
- (instancetype)initWithData:(NSData *)data
                        rows:(NSInteger)rows
                     columns:(NSInteger)columns {
    self = [super init];
    if (self == nil) {
        return self;
    }

    _data = [data mutableCopy];
    _rows = rows;
    _columns = columns;

    return self;
}

/// Creates a matrix from a Metal tensor.
///
/// - Parameter tensor: A tensor with a rank of `2` and a data type of `MTLTensorDataTypeFloat16`.
///
/// - Returns: A matrix if the method succeeds; otherwise, `nil`.
- (nullable instancetype)initFromTensor:(id<MTLTensor>)tensor {
    if (tensor.dimensions.rank != 2) {
        NSLog(@"Tensor rank isn't 2, which means it's not a 2D matrix.");
        return nil;
    }

    if (tensor.dataType != MTLTensorDataTypeFloat16) {
        NSLog(@"Tensor data type isn't `Float16`.");
        return nil;
    }

    // Metal tensors store with innermost dimension first.
    MTLTensorExtents *dimensions = tensor.dimensions;
    NSInteger columns = [dimensions extentAtDimensionIndex:0];
    NSInteger rows = [dimensions extentAtDimensionIndex:1];

    NSInteger elementCount = [Matrix totalElementsForDimensions:dimensions];
    NSInteger dataLength = elementCount * sizeof(half);
    NSMutableData *matrixData = [NSMutableData dataWithLength:dataLength];

    // Copy data from the tensor.
    MTLTensorExtents *zeroExtents = [Matrix tensorSliceOriginForRank:dimensions.rank];
    MTLTensorExtents *strides = [Matrix tensorStridesForDimensions:dimensions];

    [tensor getBytes:matrixData.mutableBytes
             strides:strides
     fromSliceOrigin:zeroExtents
     sliceDimensions:dimensions];

    return [self initWithData:matrixData rows:rows columns:columns];
}

// MARK: - Public methods

/// Multiplies this matrix with another matrix.
///
/// - Parameter multiplier: Another matrix the method multiplies with this matrix, the multiplicand.
///
/// - Returns: A product matrix if the method succeeds; otherwise, `nil`.
///
/// The method clears all negative values in the product matrix's elements to zero
/// to replicate the kernel function's rectified linear unit (ReLU) activation function.
- (Matrix *)multiplyWithMatrix:(Matrix *)multiplier {
    if (self.columns != multiplier.rows) {
        NSLog(@"Can't multiply with another matrix that doesn't have %ld rows.", self.columns);
        return nil;
    }

    const NSInteger productRows = self.rows;
    const NSInteger productColumns = multiplier.columns;
    const NSInteger productSize = sizeof(half) * productRows * productColumns;

    NSMutableData *productData = [NSMutableData dataWithLength:productSize];

    // Define strides for row-major indexing.
    const NSInteger thisStride = self.columns;
    const NSInteger thatStride = multiplier.columns;
    const NSInteger productStride = productColumns;

    /// A helper block that converts 2D coordinates and a stride to a 1D array index.
    NSInteger (^indexOf)(NSInteger row, NSInteger col, NSInteger stride) =
    ^NSInteger(NSInteger row, NSInteger col, NSInteger stride) {
        return (row * stride) + col;
    };

    // Define the pointers to each array.
    const half *thisArray = (const half *)self.data.bytes;
    const half *thatArray = (const half *)multiplier.data.bytes;
    half *productArray = (half *)productData.mutableBytes;

    // Multiply corresponding elements in row-major order.
    for (NSInteger row = 0; row < productRows; row++) {
        for (NSInteger column = 0; column < productColumns; column++) {
            float sum = 0.0f;
            for (NSInteger inner = 0; inner < self.columns; inner++) {
                NSInteger thisIndex = indexOf(row, inner, thisStride);
                NSInteger thatIndex = indexOf(inner, column, thatStride);

                sum += (float)thisArray[thisIndex] * (float)thatArray[thatIndex];
            }

            NSInteger productIndex = indexOf(row, column, productStride);

            // Set product sums with a negative value to zero.
            productArray[productIndex] = (half)fmaxf(sum, 0.0f);
        }
    }

    Matrix *productMatrix = [[Matrix alloc] initWithData:productData
                                                    rows:productRows
                                                 columns:productColumns];

    return productMatrix;
}

/// Compares this matrix with another matrix.
///
/// - Parameter other: A matrix the method compares with this matrix.
///
/// - Returns: A Boolean value that indicates whether the matrices are equal within tolerance.
///
/// The method uses a tolerance of `0.05` for 16-bit floating-point comparisons.
- (BOOL)isEqualToMatrix:(Matrix *)other {
    const float tolerance = 0.05f;

    if (self.rows != other.rows || self.columns != other.columns) {
        NSLog(@"Matrix dimensions don't match.");
        NSLog(@"This matrix: (%ld x %ld)", self.rows, self.columns);
        NSLog(@"That matrix: (%ld x %ld)", other.rows, other.columns);
        return NO;
    }

    const half *thisMatrix = (const half *)self.data.bytes;
    const half *thatMatrix = (const half *)other.data.bytes;
    NSInteger elementCount = self.rows * self.columns;

    for (NSInteger index = 0; index < elementCount; index++) {
        float diff = fabsf((float)thisMatrix[index] - (float)thatMatrix[index]);
        if (diff > tolerance) {
            NSLog(@"Mismatch at element %ld.", index);
            NSLog(@"This matrix: %.6f", (float)thisMatrix[index]);
            NSLog(@"That matrix: %.6f", (float)thatMatrix[index]);
            NSLog(@"Difference: %.6f", diff);
            return NO;
        }
    }

    return YES;
}

/// Prints a preview of the matrix.
///
/// - Parameter name: A name for the matrix, which the method prints above its data.
///
/// The method only prints up to the first four rows and columns.
- (void)printWithName:(NSString *)name {
    const half *matrix = (const half *)self.data.bytes;

    NSInteger maxRowsToPrint = MIN(self.rows, 4);
    NSInteger maxColsToPrint = MIN(self.columns, 4);

    printf("%s (%ld x %ld):\n", [name UTF8String], self.rows, self.columns);

    for (NSInteger row = 0; row < maxRowsToPrint; row++) {
        for (NSInteger column = 0; column < maxColsToPrint; column++) {
            printf("%7.3f ", (float)matrix[row * self.columns + column]);
        }
        if (self.columns > 4) {
            printf("...");
        }
        printf("\n");
    }

    if (self.rows > 4) {
        // Print a row of aligned "..." for each column.
        for (NSInteger column = 0; column < maxColsToPrint; column++) {
            printf("%7s ", "...");
        }
        printf("\n");
    }

    printf("\n");
}

/// Copies this matrix's data into a Metal tensor.
///
/// - Parameter tensor: A tensor that receives the matrix data.
- (void)copyDataToTensor:(id<MTLTensor>)tensor {
    if (tensor.dimensions.rank != 2) {
        NSLog(@"Tensor rank isn't 2, which means it's not a 2D matrix.");
        return;
    }

    MTLTensorExtents *dimensions = tensor.dimensions;
    MTLTensorExtents *zeroExtents = [Matrix tensorSliceOriginForRank:dimensions.rank];
    MTLTensorExtents *strides = [Matrix tensorStridesForDimensions:dimensions];

    [tensor replaceSliceOrigin:zeroExtents
               sliceDimensions:dimensions
                     withBytes:self.data.bytes
                       strides:strides];
}

// MARK: - Helper methods

/// Fills the matrix with random floating-point values.
///
/// Each element is in the range `[-1.0, 1.0]`.
- (void)fillWithRandomData {
    half *elements = (half *)((NSMutableData *)_data).mutableBytes;
    NSInteger elementCount = _rows * _columns;

    for (NSInteger index = 0; index < elementCount; index++) {
        float randomFloat = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        elements[index] = (half)randomFloat;
    }
}

@end
