/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Implements a matrix class that multiplies matrices on the CPU for verification.
*/

#import "Matrix.h"
#import "Matrix+TensorUtilities.h"

@interface Matrix () {
    /// The matrix data as a contiguous array of floats.
    NSMutableData *_data;
}
@end

@implementation Matrix

- (NSData *)data {
    return _data;
}

// MARK: - Factory methods

/// Creates a matrix with random floating-point values.
/// - Parameters:
///   - rows: The number of rows in the new matrix.
///   - columns: The number of columns in the new matrix.
+ (instancetype)matrixWithRandomRows:(NSInteger)rows
                             columns:(NSInteger)columns {
    Matrix *matrix = [[self alloc] initWithRows:rows columns:columns];
    [matrix fillWithRandomData];
    return matrix;
}

// MARK: - Initializers

/// Creates a matrix with every element equal to zero.
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
    _data = [NSMutableData dataWithLength:(rows * columns) * sizeof(Float32)];

    return self;
}

/// Creates a matrix with existing data.
/// - Parameters:
///   - data: Matrix data.
///   - rows: The number of rows in the matrix.
///   - columns: The number of columns in the matrix.
- (instancetype)initWithData:(NSData *)data
                        rows:(NSInteger)rows
                     columns:(NSInteger)columns {
    self = [super init];
    if (self == nil) {
        return self;
    }

    _data = [data copy];
    _rows = rows;
    _columns = columns;

    return self;
}

/// Creates a matrix from a Metal tensor.
/// - Parameter tensor: A tensor that has a rank of `2` and a data type equal to
/// `MTLTensorDataTypeFloat32`.
/// - Returns: A matrix if the method succeeds; otherwise, `nil`.
- (nullable instancetype)initFromTensor:(id<MTLTensor>)tensor {
    if (tensor.dimensions.rank != 2) {
        NSLog(@"Tensor rank isn't 2, which means it's not a 2D matrix.");
        return nil;
    }

    if (tensor.dataType != MTLTensorDataTypeFloat32) {
        NSLog(@"Tensor data type isn't `Float32`.");
        return nil;
    }

    // Metal tensors define their dimensions with the innermost dimension first.
    MTLTensorExtents *dimensions = tensor.dimensions;
    NSInteger columns = [dimensions extentAtDimensionIndex:0];
    NSInteger rows = [dimensions extentAtDimensionIndex:1];

    NSInteger elementCount = [Matrix totalElementsForDimensions:dimensions];
    NSInteger dataLength = elementCount * sizeof(Float32);
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
/// - Parameter other: A matrix the method multiplies with this matrix.
/// - Returns: A product matrix if the method succeeds; otherwise, `nil`.
- (Matrix *)multiplyWithMatrix:(Matrix *)other {
    if (self.columns != other.rows) {
        NSLog(@"Can't multiply with another matrix that doesn't have %ld rows.", self.columns);
        return nil;
    }

    const NSInteger productRows = self.rows;
    const NSInteger productColumns = other.columns;
    const NSInteger productSize = sizeof(Float32) * productRows * productColumns;

    NSMutableData *productData = [NSMutableData dataWithLength:productSize];

    // Define strides for row-major indexing.
    const NSInteger thisStride = self.columns;
    const NSInteger thatStride = other.columns;
    const NSInteger productStride = productColumns;

    /// A helper block that converts 2D coordinates and a stride to a 1D array index.
    NSInteger (^indexOf)(NSInteger row, NSInteger col, NSInteger stride) =
    ^NSInteger(NSInteger row, NSInteger col, NSInteger stride) {
        return (row * stride) + col;
    };

    // Define the pointers to each array.
    const Float32 *thisArray = (const Float32 *)self.data.bytes;
    const Float32 *thatArray = (const Float32 *)other.data.bytes;
    Float32 *productArray = (Float32 *)productData.mutableBytes;

    // Multiply corresponding elements in row-major order.
    for (NSInteger row = 0; row < productRows; row++) {
        for (NSInteger column = 0; column < productColumns; column++) {
            Float32 sum = 0.0f;
            for (NSInteger inner = 0; inner < self.columns; inner++) {
                NSInteger thisIndex = indexOf(row, inner, thisStride);
                NSInteger thatIndex = indexOf(inner, column, thatStride);

                sum += thisArray[thisIndex] * thatArray[thatIndex];
            }

            NSInteger productIndex = indexOf(row, column, productStride);
            productArray[productIndex] = sum;
        }
    }

    Matrix *productMatrix = [[Matrix alloc] initWithData:productData
                                                    rows:productRows
                                                 columns:productColumns];

    return productMatrix;
}

/// Compares this matrix with another matrix.
///
/// The method applies a tolerance of `1e-5`, which means it considers two values
/// equal if their difference's magnitude is less than or equal to the tolerance value.
/// - Parameter other: A matrix the method compares with this matrix.
/// - Returns: A Boolean value that indicates whether the matrices are equal within a small tolerance.
- (BOOL)isEqualToMatrix:(Matrix *)other {
    const Float32 tolerance = 1e-5f;

    if (self.rows != other.rows || self.columns != other.columns) {
        NSLog(@"Matrix dimensions don't match.");
        NSLog(@"This matrix: (%ld x %ld)", self.rows, self.columns);
        NSLog(@"That matrix: (%ld x %ld)", other.rows, other.columns);
        return NO;
    }

    const Float32 *thisMatrix = (const Float32 *)self.data.bytes;
    const Float32 *thatMatrix = (const Float32 *)other.data.bytes;
    NSInteger elementCount = self.rows * self.columns;

    for (NSInteger index = 0; index < elementCount; index++) {
        Float32 diff = fabsf(thisMatrix[index] - thatMatrix[index]);
        if (diff > tolerance) {
            NSLog(@"Mismatch at element %ld.", index);
            NSLog(@"This matrix: %.6f", thisMatrix[index]);
            NSLog(@"That matrix: %.6f", thatMatrix[index]);
            NSLog(@"Difference: %.6f", diff);
            return NO;
        }
    }

    return YES;
}

/// Prints the matrix to the console.
/// - Parameter name: A name for the matrix the method prints above its data.
- (void)printWithName:(NSString *)name {
    const Float32 *matrix = (const Float32 *)self.data.bytes;

    printf("%s (%ld x %ld):\n", [name UTF8String], self.rows, self.columns);

    for (NSInteger row = 0; row < self.rows; row++) {
        for (NSInteger column = 0; column < self.columns; column++) {
            printf("%.3f ", matrix[row * self.columns + column]);
        }
        printf("\n");
    }

    printf("\n");
}

/// Copies this matrix's data into a Metal tensor.
/// - Parameter tensor: A tensor that receives the matrix data.
- (void)copyDataToTensor:(id<MTLTensor>)tensor {
    if (tensor.dimensions.rank != 2) {
        NSLog(@"The tensor rank isn't 2, which means it doesn't represent a 2D matrix.");
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

/// Fills the matrix with random floating-point values between zero and one.
///
/// The method generates values in the range `[0.0, 1.0]`.
- (void)fillWithRandomData {
    Float32 *floatPtr = (Float32 *)((NSMutableData *)_data).mutableBytes;
    NSInteger elementCount = _rows * _columns;

    for (NSInteger index = 0; index < elementCount; index++) {
        floatPtr[index] = (Float32)rand() / (Float32)(RAND_MAX);
    }
}

@end
