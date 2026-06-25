/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Methods that set up tensors for the machine learning pipeline.
*/

#import "MLMatrixMultiplier.h"

NS_ASSUME_NONNULL_BEGIN

typedef NSDictionary<NSString *, id<MTLTensor>> TensorsByName;
typedef NSDictionary<NSString *, id<MTLTensorBinding>> TensorBindingsByName;

@interface MLMatrixMultiplier (TensorSetup)

/// Extracts tensor bindings from pipeline reflection.
/// - Parameter pipelineState: A machine learning-pipeline state with tensor bindings.
/// - Returns: A dictionary mapping tensor names to bindings if the method succeeds; otherwise, `nil`.
- (nullable TensorBindingsByName *)
extractTensorBindingsFromPipelineState:(id<MTL4MachineLearningPipelineState>)pipelineState;

/// Creates tensors for a dictionary of bindings.
/// - Parameters:
///   - bindings: A dictionary mapping tensor names to bindings.
///   - device: A Metal device that creates the tensors.
/// - Returns: A dictionary mapping tensor names to tensors if the method succeeds; otherwise, `nil`.
- (nullable TensorsByName *)
createTensorsForBindings:(TensorBindingsByName *)bindings
              withDevice:(id<MTLDevice>)device;

@end

NS_ASSUME_NONNULL_END
