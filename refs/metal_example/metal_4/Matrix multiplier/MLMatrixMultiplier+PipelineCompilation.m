/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Compiles the machine learning pipeline state.
*/

#import "MLMatrixMultiplier+PipelineCompilation.h"

@implementation MLMatrixMultiplier (PipelineCompilation)

/// Compiles a pipeline state based on an array with two matrices.
/// - Parameters:
///   - compiler: A compiler that creates the pipeline state.
///   - library: A library that contains the machine learning function.
///   - matrices: An array with two matrices.
/// - Returns: A pipeline state if the method succeeds; otherwise, `nil`.
- (nullable id<MTL4MachineLearningPipelineState>)
createPipelineStateWithCompiler:(id<MTL4Compiler>)compiler
                    fromLibrary:(id<MTLLibrary>)library
                    forMatrices:(NSArray<Matrix *> *)matrices {

    // Retrieve the metadata for the library's main function.
    MTLFunctionReflection *functionReflection;
    functionReflection = [library reflectionForFunctionWithName:@"main"];
    if (functionReflection == nil) {
        NSLog(@"Can't get function reflection for the main function.");
        return nil;
    }

    MTL4LibraryFunctionDescriptor *functionDescriptor =
    [MTL4LibraryFunctionDescriptor new];
    functionDescriptor.name = @"main";
    functionDescriptor.library = library;

    MTL4MachineLearningPipelineDescriptor *pipelineDescriptor;
    pipelineDescriptor = [MTL4MachineLearningPipelineDescriptor new];
    pipelineDescriptor.machineLearningFunctionDescriptor = functionDescriptor;

    // Enable reflection to get binding information after compilation.
    MTL4PipelineOptions *options = [MTL4PipelineOptions new];
    options.shaderReflection = MTL4ShaderReflectionBindingInfo;
    pipelineDescriptor.options = options;

    // Collect input tensor bindings and sort by name.
    NSMutableArray<id<MTLTensorBinding>> *inputBindings = [NSMutableArray new];
    for (id<MTLBinding> binding in functionReflection.bindings) {
        if (binding.type != MTLBindingTypeTensor) {
            continue;
        }

        id<MTLTensorBinding> tensorBinding = (id<MTLTensorBinding>)binding;

        // Collect the input bindings and ignore any output bindings.
        if (![[tensorBinding.name lowercaseString] containsString:@"input"]) {
            continue;
        }

        // Ignore bindings that don't work with 2D tensors.
        if (tensorBinding.dimensions.rank != 2) {
            continue;
        }

        [inputBindings addObject:tensorBinding];
    }

    if (inputBindings.count != 2) {
        NSLog(@"Expected %ld input tensors, but found %ld.", matrices.count, inputBindings.count);
        return nil;
    }

    // Sort input bindings by name.
    [inputBindings sortUsingComparator:^NSComparisonResult(id<MTLTensorBinding> tensorBindingA,
                                                           id<MTLTensorBinding> tensorBindingB) {
        return [tensorBindingA.name compare:tensorBindingB.name];
    }];

    // Set the input dimensions for each tensor binding from the matrices.
    for (NSInteger index = 0; index < inputBindings.count; index++) {
        id<MTLTensorBinding> tensorBinding = inputBindings[index];
        Matrix *matrix = matrices[index];

        NSInteger dimensions[] = {matrix.columns, matrix.rows};
        MTLTensorExtents *extents = [[MTLTensorExtents alloc] initWithRank:2
                                                                    values:dimensions];

        [pipelineDescriptor setInputDimensions:extents
                                 atBufferIndex:tensorBinding.index];
    }

    NSError *error = nil;
    id<MTL4MachineLearningPipelineState> state =
    [compiler newMachineLearningPipelineStateWithDescriptor:pipelineDescriptor
                                                      error:&error];

    if (state == nil) {
        NSLog(@"Can't create a machine learning-pipeline state.");
        NSLog(@"Details: %@", error);
        return nil;
    }

    return state;
}

@end
