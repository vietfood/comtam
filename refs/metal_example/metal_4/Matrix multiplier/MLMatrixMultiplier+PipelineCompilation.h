/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Methods that compile the machine learning pipeline state.
*/

#import "MLMatrixMultiplier.h"

NS_ASSUME_NONNULL_BEGIN

@interface MLMatrixMultiplier (PipelineCompilation)

/// Compiles a pipeline state based on an array with two matrices.
/// - Parameters:
///   - compiler: A compiler that creates the pipeline state.
///   - library: A library that contains the machine learning function.
///   - matrices: An array with two matrices.
/// - Returns: A pipeline state if the method succeeds; otherwise, `nil`.
- (nullable id<MTL4MachineLearningPipelineState>)
createPipelineStateWithCompiler:(id<MTL4Compiler>)compiler
                    fromLibrary:(id<MTLLibrary>)library
                    forMatrices:(NSArray<Matrix *> *)matrices;

@end

NS_ASSUME_NONNULL_END
