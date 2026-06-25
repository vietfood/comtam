/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Sets up tensors for the machine learning pipeline.
*/

#import "MLMatrixMultiplier+TensorSetup.h"

@implementation MLMatrixMultiplier (TensorSetup)

/// Finds the tensor bindings for a pipeline state and adds them to a dictionary.
/// - Parameter pipelineState: A machine learning-pipeline state with tensor bindings.
/// - Returns: A dictionary that maps each tensor binding to its name if the method succeeds; otherwise, `nil`.
- (nullable TensorBindingsByName *)
extractTensorBindingsFromPipelineState:(id<MTL4MachineLearningPipelineState>)pipelineState {
    NSMutableDictionary<NSString *, id<MTLTensorBinding>> *bindingsByName;
    bindingsByName = [NSMutableDictionary new];

    for (id<MTLBinding> binding in pipelineState.reflection.bindings) {
        if (binding.type != MTLBindingTypeTensor) {
            continue;
        }

        bindingsByName[binding.name] = (id<MTLTensorBinding>)binding;
    }

    return bindingsByName;
}

/// Creates tensors for each tensor binding and adds them to a dictionary.
/// - Parameters:
///   - bindings: A dictionary that maps tensor names to their bindings.
///   - device: A Metal device the method needs to create the tensor instances.
/// - Returns: A dictionary that maps each tensor instance to its name if the method succeeds; otherwise, `nil`.
- (nullable TensorsByName *)createTensorsForBindings:(TensorBindingsByName *)bindings
                                           withDevice:(id<MTLDevice>)device {
    NSMutableDictionary<NSString *, id<MTLTensor>> *tensorsByName;
    tensorsByName = [NSMutableDictionary new];

    MTLTensorDescriptor *tensorDescriptor = [MTLTensorDescriptor new];
    tensorDescriptor.usage = MTLTensorUsageMachineLearning;

    for (NSString *name in bindings) {
        id<MTLTensorBinding> binding = bindings[name];
        MTLTensorExtents *dimensions = binding.dimensions;
        MTLTensorDataType dataType = binding.tensorDataType;

        // Validate static tensor shapes (no dynamic dimensions allowed).
        if (dimensions == nil) {
            NSLog(@"The app doesn't support unranked tensors.");
            return nil;
        }

        /// A sentinel value that indicates a dimension has a dynamic shape.
        const NSUInteger sentinelValueForVariableDimensions = -1;

        // Return early if any dimension has a dynamic shape.
        for (NSUInteger index = 0; index < dimensions.rank; index++) {
            if ([dimensions extentAtDimensionIndex:index] == sentinelValueForVariableDimensions) {
                NSLog(@"The app doesn't support dynamic tensor shapes.");
                return nil;
            }
        }

        tensorDescriptor.dimensions = dimensions;
        tensorDescriptor.dataType = dataType;

        NSError *error = nil;
        id<MTLTensor> tensor = [device newTensorWithDescriptor:tensorDescriptor
                                                         error:&error];
        if (tensor == nil) {
            NSLog(@"Can't create tensor.");
            NSLog(@"Details: %@", error);
            return nil;
        }

        tensorsByName[name] = tensor;
    }

    return tensorsByName;
}

@end
