/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Implements a class that multiplies matrices with a cooperative tensor in shader code.
*/

#import "MatrixMultiplier.h"

@implementation MatrixMultiplier {
    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTL4CommandQueue> queue;
    id<MTL4CommandAllocator> commandAllocator;
    id<MTLComputePipelineState> pipelineState;
    id<MTL4ArgumentTable> argumentTable;
    id<MTLResidencySet> residencySet;
    id<MTLSharedEvent> event;
    NSUInteger signaledValue;

    NSArray<id<MTLTensor>> *inputTensors;
    id<MTLTensor> outputTensor;
}

// MARK: - Initializers

/// Creates a matrix multiplier.
- (instancetype)init {
    self = [super init];

    if (self) {
        device = MTLCreateSystemDefaultDevice();

        if (![device supportsFamily:MTLGPUFamilyApple7]) {
            NSLog(@"Device must support GPU Family Apple7");
            self = nil;
            return nil;
        }

        library = [device newDefaultLibrary];
        queue = [device newMTL4CommandQueue];
        commandAllocator = [device newCommandAllocator];

        // Compile the matrix multiplication pipeline state.
        pipelineState = [self createPipelineState];
        if (!pipelineState) {
            self = nil;
            return nil;
        }

        // Create a shared event to wait for the command buffer to complete.
        event = [device newSharedEvent];
        signaledValue = 0;
    }

    return self;
}

// MARK: - Configuration

/// Prepares the multiplier to run a matrix multiplication with cooperative tensors
/// on a Metal device.
/// - Parameters:
///   - matrix1: A matrix for the first input.
///   - matrix2: A matrix for the second input.
/// - Returns: A Boolean value that indicates whether the method succeeds.
- (BOOL)configureWithMatrix1:(Matrix *)matrix1
                     matrix2:(Matrix *)matrix2 {
    // Validate the dimensions.
    if (matrix1.columns != matrix2.rows) {
        NSLog(@"Can't multiply matrices: matrix1 columns (%ld) != matrix2 rows (%ld)",
              matrix1.columns, matrix2.rows);
        return NO;
    }

    // Create the tensors.
    id<MTLTensor> tensorA = [self create2DTensorWithDataType:MTLTensorDataTypeFloat16
                                                       height:matrix1.rows
                                                        width:matrix1.columns];
    id<MTLTensor> tensorB = [self create2DTensorWithDataType:MTLTensorDataTypeFloat16
                                                       height:matrix2.rows
                                                        width:matrix2.columns];
    outputTensor = [self create2DTensorWithDataType:MTLTensorDataTypeFloat16
                                             height:matrix1.rows
                                              width:matrix2.columns];

    if (!tensorA || !tensorB || !outputTensor) {
        return NO;
    }

    inputTensors = @[tensorA, tensorB];

    // Copy matrix data to the tensors.
    [matrix1 copyDataToTensor:inputTensors[0]];
    [matrix2 copyDataToTensor:inputTensors[1]];

    // Create an argument table to bind the input and output tensors.
    MTL4ArgumentTableDescriptor *argumentTableDescriptor = [[MTL4ArgumentTableDescriptor alloc] init];
    argumentTableDescriptor.maxBufferBindCount = 3;

    NSError *error = nil;
    argumentTable = [device newArgumentTableWithDescriptor:argumentTableDescriptor error:&error];

    if (!argumentTable) {
        NSLog(@"Error creating argument table: %@", error);
        return NO;
    }

    // Create a residency set to track which resources need to be resident during runtime.
    MTLResidencySetDescriptor *residencyDescriptor = [[MTLResidencySetDescriptor alloc] init];
    residencyDescriptor.initialCapacity = 3;

    residencySet = [device newResidencySetWithDescriptor:residencyDescriptor error:&error];

    if (!residencySet) {
        NSLog(@"Error creating residency set: %@", error);
        return NO;
    }

    [residencySet addAllocation:inputTensors[0]];
    [residencySet addAllocation:inputTensors[1]];
    [residencySet addAllocation:outputTensor];
    [residencySet commit];

    // Bind the tensors to the argument table.
    [argumentTable setResource:inputTensors[0].gpuResourceID atBufferIndex:0];
    [argumentTable setResource:inputTensors[1].gpuResourceID atBufferIndex:1];
    [argumentTable setResource:outputTensor.gpuResourceID atBufferIndex:2];

    return YES;
}

// MARK: - Execution

/// Encodes a compute pass and runs it on the GPU.
- (void)encodeAndRunComputation {
    // Create a command buffer and begin encoding commands into it.
    id<MTL4CommandBuffer> commandBuffer = [device newCommandBuffer];
    [commandBuffer beginCommandBufferWithAllocator:commandAllocator];

    // Add the residency set to the command buffer so that the resources are
    // resident when running the command buffer.
    [commandBuffer useResidencySet:residencySet];

    // Create a compute command encoder.
    id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    // Set the pipeline state and the argument table.
    [encoder setComputePipelineState:pipelineState];
    [encoder setArgumentTable:argumentTable];

    // Calculate the threadgroup configuration.
    NSInteger tileSizeY = 64;
    NSInteger tileSizeX = 64;
    NSInteger numSIMDGroups = 4;

    // Get output dimensions from the tensor.
    NSInteger outputWidth = [outputTensor.dimensions extentAtDimensionIndex:0];
    NSInteger outputHeight = [outputTensor.dimensions extentAtDimensionIndex:1];

    MTLSize threadgroups = MTLSizeMake((outputWidth + tileSizeX - 1) / tileSizeX,
                                       (outputHeight + tileSizeY - 1) / tileSizeY,
                                       1);

    NSUInteger simdGroupSize = pipelineState.threadExecutionWidth;

    // Dispatch a threadgroup for each tile of the output tensor.
    [encoder dispatchThreadgroups:threadgroups
            threadsPerThreadgroup:MTLSizeMake(numSIMDGroups * simdGroupSize, 1, 1)];

    // End encoding.
    [encoder endEncoding];
    [commandBuffer endCommandBuffer];

    // Commit the command buffer.
    [queue commit:&commandBuffer count:1];

    signaledValue++;

    // Signal the shared event when the command buffer completes.
    [queue signalEvent:event value:signaledValue];

    // Wait for the event to signal.
    [event waitUntilSignaledValue:signaledValue timeoutMS:1000];
}

// MARK: - Accessors

/// Returns the product tensor from the GPU computation.
- (id<MTLTensor>)productTensor {
    return outputTensor;
}

// MARK: - Helper methods

/// Creates a pipeline state for the matrix multiplication kernel.
- (id<MTLComputePipelineState>)createPipelineState {
    MTLComputePipelineDescriptor *pipelineDescriptor = [[MTLComputePipelineDescriptor alloc] init];

    pipelineDescriptor.computeFunction = [library newFunctionWithName:@"matrix_multiplication_kernel"];
    pipelineDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;

    NSError *error = nil;

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:pipelineDescriptor
                                                                                  options:0
                                                                               reflection:nil
                                                                                    error:&error];

    if (!pipeline) {
        NSLog(@"Error creating pipeline state: %@", error);
    }

    return pipeline;
}

/// Creates a 2D tensor with the specified data type and dimensions.
- (id<MTLTensor>)create2DTensorWithDataType:(MTLTensorDataType)dataType
                                     height:(NSInteger)height
                                      width:(NSInteger)width
{
    NSInteger dimensions[] = { width, height };

    MTLTensorDescriptor *tensorDescriptor = [[MTLTensorDescriptor alloc] init];

    tensorDescriptor.dataType = dataType;
    tensorDescriptor.dimensions = [[MTLTensorExtents alloc] initWithRank:2 values:dimensions];
    tensorDescriptor.usage = MTLTensorUsageCompute;

    NSError *error = nil;

    id<MTLTensor> tensor = [device newTensorWithDescriptor:tensorDescriptor
                                                      error:&error];

    if (!tensor) {
        NSLog(@"Error creating tensor: %@", error);
    }

    return tensor;
}

@end
