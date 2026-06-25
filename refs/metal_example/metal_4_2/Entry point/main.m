/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Multiplies matrices on a GPU and verifies the product against a CPU implementation.
*/

#import "MatrixMultiplier.h"
#import "Matrix.h"
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Creates a matrix with random data.
/// - Parameters:
///   - rows: The number of rows in the new matrix.
///   - columns: The number of columns in the new matrix.
static Matrix *createRandomMatrix(NSInteger rows, NSInteger columns) {
    return [Matrix matrixWithRandomRows:rows columns:columns];
}

/// Multiplies two matrices on the CPU.
/// - Parameters:
///   - matrix1: A matrix for the first operand.
///   - matrix2: A matrix for the second operand.
/// - Returns: A product matrix if the method succeeds; otherwise, `nil`.
static Matrix * _Nullable multiplyMatricesOnCPU(Matrix *matrix1,
                                                Matrix *matrix2) {
    Matrix *product = [matrix1 multiplyWithMatrix:matrix2];
    if (product == nil) {
        NSLog(@"Can't multiply the matrices on the CPU.");
    }
    return product;
}

/// Multiplies two matrices on the GPU using Metal.
/// - Parameters:
///   - matrix1: A matrix for the first operand.
///   - matrix2: A matrix for the second operand.
/// - Returns: A product matrix if the method succeeds; otherwise, `nil`.
static Matrix * _Nullable multiplyMatricesOnGPU(Matrix *matrix1,
                                                Matrix *matrix2) {
    MatrixMultiplier *multiplier = [MatrixMultiplier new];
    if (multiplier == nil) {
        NSLog(@"Can't create a matrix multiplier.");
        return nil;
    }

    BOOL success = [multiplier configureWithMatrix1:matrix1 matrix2:matrix2];
    if (!success) {
        NSLog(@"Can't configure the matrix multiplier.");
        return nil;
    }

    [multiplier encodeAndRunComputation];
    Matrix *product = [[Matrix alloc] initFromTensor:[multiplier productTensor]];
    return product;
}

/// Compares two product matrices and prints whether they're equal.
/// - Parameters:
///   - gpuProduct: A matrix from the GPU computation.
///   - cpuProduct: A matrix from the CPU computation.
/// - Returns: A Boolean value that indicates whether the products match.
static BOOL compareResults(Matrix *gpuProduct, Matrix *cpuProduct) {
    BOOL productsMatch = [gpuProduct isEqualToMatrix:cpuProduct];
    printf("%s\n", productsMatch ? "✅ GPU == CPU" : "❌ GPU != CPU");
    return productsMatch;
}

NS_ASSUME_NONNULL_END

int main(void) {
    @autoreleasepool {
        Matrix *matrix1 = createRandomMatrix(512, 512);
        Matrix *matrix2 = createRandomMatrix(512, 512);

        [matrix1 printWithName:@"Matrix 1"];
        [matrix2 printWithName:@"Matrix 2"];

        Matrix *cpuProduct = multiplyMatricesOnCPU(matrix1, matrix2);
        Matrix *gpuProduct = multiplyMatricesOnGPU(matrix1, matrix2);

        if (cpuProduct == nil || gpuProduct == nil) {
            return EXIT_FAILURE;
        }

        [cpuProduct printWithName:@"CPU Product"];
        [gpuProduct printWithName:@"GPU Product"];

        BOOL productsMatch = compareResults(gpuProduct, cpuProduct);
        return productsMatch ? EXIT_SUCCESS : EXIT_FAILURE;
    }
}
