/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Manages Metal resources and multiplies matrices using the machine learning pipeline.
*/

#import "MLMatrixMultiplier.h"
#import "MLMatrixMultiplier+PipelineCompilation.h"
#import "MLMatrixMultiplier+TensorSetup.h"

@interface MLMatrixMultiplier () {
    /// The main API protocol that represents a GPU.
    id<MTLDevice> device;

    /// A command queue that submits work to its Metal device.
    id<MTL4CommandQueue> commandQueue;


    /// A command buffer that records machine learning commands.
    id<MTL4CommandBuffer> commandBuffer;

    /// A command allocator that provides memory for command encoding.
    id<MTL4CommandAllocator> commandAllocator;

    /// A library that contains the machine learning function.
    id<MTLLibrary> library;

    /// A compiler instance that creates machine learning-pipeline states.
    id<MTL4Compiler> compiler;

    /// A pipeline state that runs a machine learning dispatch pass.
    id<MTL4MachineLearningPipelineState> pipelineState;

    /// An array of input tensors for a machine learning pass.
    NSArray<id<MTLTensor>> *inputTensors;

    /// An output tensor for a machine learning pass.
    id<MTLTensor> outputTensor;

    /// An argument table that binds tensor arguments for a machine learning pass.
    id<MTL4ArgumentTable> argumentTable;

    /// A heap that a machine learning encoder allocates temporary resources from
    /// as it encodes a machine learning dispatch pass.
    id<MTLHeap> intermediatesHeap;

    /// A shared event that notifies the CPU when the GPU finishes running a command buffer.
    ///
    /// The app needs an ``MTLSharedEvent`` instance because it can synchronize work
    /// between the CPU and GPU, unlike an ``MTLEvent``, which can't synchronize CPU-based work.
    id<MTLSharedEvent> sharedEvent;
}
@end

@implementation MLMatrixMultiplier

/// Creates a matrix multiplier.
- (instancetype)init {
    self = [super init];
    if (self) {
        device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            NSLog(@"Can't create a Metal device.");
            return nil;
        }

        commandQueue = [device newMTL4CommandQueue];
        if (commandQueue == nil) {
            NSLog(@"Can't create the command queue.");
            return nil;
        }

        library = [self loadMetalPackage:@"matrix-multiplication"];
        if (library == nil) {
            return nil;
        }

        compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new]
                                               error:nil];

        if (compiler == nil) {
            NSLog(@"Can't create a Metal compiler.");
            return nil;
        }

        sharedEvent = [device newSharedEvent];
        if (sharedEvent == nil) {
            NSLog(@"Can't create a shared event.");
            return nil;
        }

        commandBuffer = [device newCommandBuffer];
        if (commandBuffer == nil) {
            NSLog(@"Can't create a command buffer.");
            return nil;
        }

        commandAllocator = [device newCommandAllocator];
        if (commandAllocator == nil) {
            NSLog(@"Can't create a command allocator.");
            return nil;
        }
    }
    return self;
}

/// Loads a Metal package from the app bundle.
/// - Parameter name: The package name without the file extension.
/// - Returns: A library instance if the method succeeds; otherwise, `nil`.
- (nullable id<MTLLibrary>)loadMetalPackage:(NSString *)name {
    NSURL *packageURL =
    [[NSBundle mainBundle] URLForResource:name
                            withExtension:@"mtlpackage"];
    if (packageURL == nil) {
        NSLog(@"Can't find %@.mtlpackage in the app's bundle.", name);
        return nil;
    }

    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:packageURL error:&error];
    if (library == nil) {
        NSLog(@"Can't load Metal library at URL: %@.", packageURL);
        NSLog(@"Details: %@", error);
        return nil;
    }

    return library;
}

/// Prepares the multiplier to run a matrix multiplication in a machine learning
/// pass on a Metal device.
/// - Parameters:
///   - matrix1: A matrix for the first input.
///   - matrix2: A matrix for the second input.
/// - Returns: A Boolean value that indicates whether the method succeeds.
- (BOOL)configureWithMatrix1:(Matrix *)matrix1 matrix2:(Matrix *)matrix2 {
    if (matrix1.columns != matrix2.rows) {
        NSLog(@"Matrix dimensions are incompatible for multiplication.");
        return NO;
    }

    NSArray<Matrix *> *matrices = @[matrix1, matrix2];

    // Create pipeline state from matrix dimensions.
    pipelineState = [self createPipelineStateWithCompiler:compiler
                                              fromLibrary:library
                                              forMatrices:matrices];
    if (pipelineState == nil) {
        return NO;
    }

    // Extract tensor bindings from pipeline reflection.
    TensorBindingsByName *bindingsByName =
    [self extractTensorBindingsFromPipelineState:pipelineState];
    if (bindingsByName == nil) {
        return NO;
    }

    // Create tensors for the bindings.
    TensorsByName *tensorsByName;
    tensorsByName = [self createTensorsForBindings:bindingsByName
                                        withDevice:device];
    if (tensorsByName == nil) {
        return NO;
    }

    // Assign tensors by name from the dictionary.
    inputTensors = @[tensorsByName[@"inputA"], tensorsByName[@"inputB"]];
    outputTensor = tensorsByName[@"output"];
    if (nil == outputTensor || nil == inputTensors[0] || nil == inputTensors[1]) {

        NSLog(@"Can't find all the input and output bindings.");
        return NO;
    }

    // Fill in the input tensors with data from a matrix.
    for (NSInteger index = 0; index < matrices.count; index++) {
        Matrix *matrix = matrices[index];
        id<MTLTensor> tensor = inputTensors[index];
        [matrix copyDataToTensor:tensor];
    }

    // Create an argument table.
    MTL4ArgumentTableDescriptor *argumentTableDescriptor =
    [MTL4ArgumentTableDescriptor new];
    // Configure the table to store bindings for two input tensors and one output tensor.
    argumentTableDescriptor.maxBufferBindCount = 3;
    argumentTableDescriptor.initializeBindings = YES;

    NSError *error = nil;
    argumentTable = [device newArgumentTableWithDescriptor:argumentTableDescriptor
                                                     error:&error];
    if (argumentTable == nil) {
        NSLog(@"Can't create argument table.");
        NSLog(@"Details: %@", error);
        return NO;
    }

    // Bind tensors to the argument table.
    for (NSString *tensorName in bindingsByName) {
        id<MTLTensorBinding> binding = bindingsByName[tensorName];
        id<MTLTensor> tensor = tensorsByName[tensorName];

        [argumentTable setResource:tensor.gpuResourceID
                     atBufferIndex:binding.index];
    }

    // Create an intermediates heap.
    MTLHeapDescriptor *heapDescriptor = [MTLHeapDescriptor new];
    heapDescriptor.type = MTLHeapTypePlacement;
    heapDescriptor.size = pipelineState.intermediatesHeapSize;

    intermediatesHeap = [device newHeapWithDescriptor:heapDescriptor];
    if (intermediatesHeap == nil) {
        NSLog(@"Can't create a heap for intermediates.");
        return NO;
    }

    return YES;
}

/// Encodes a machine learning pass and runs it on the GPU.
- (void)encodeAndRunModelInference {
    [commandBuffer beginCommandBufferWithAllocator:commandAllocator];

    id<MTL4MachineLearningCommandEncoder> encoder;
    encoder = [commandBuffer machineLearningCommandEncoder];

    [encoder setArgumentTable:argumentTable];
    [encoder setPipelineState:pipelineState];
    [encoder dispatchNetworkWithIntermediatesHeap:intermediatesHeap];
    [encoder endEncoding];

    [commandBuffer endCommandBuffer];

    [commandQueue commit:&commandBuffer count:1];

    // Add a command to the queue that increments the shared event's value.
    uint64_t signalValue = sharedEvent.signaledValue + 1;
    [commandQueue signalEvent:sharedEvent value:signalValue];

    /// A timeout duration in milliseconds for the machine learning pass.
    ///
    /// A good timeout gives the GPU adequate time to run the pass without stalling the app indefinitely.
    /// A 100 milliseconds duration is enough time for the model to run a matrix
    /// multiplication inference and can also quickly detect GPU errors or stalls.
    const uint64_t kMLPassTimeoutMilliseconds = 100;

    // Wait for the GPU to complete the work.
    BOOL success = [sharedEvent waitUntilSignaledValue:signalValue
                                             timeoutMS:kMLPassTimeoutMilliseconds];
    if (!success) {
        NSLog(@"The machine learning pass timed out.");
    }
}

/// Returns the product tensor from the GPU computation.
- (id<MTLTensor>)productTensor {
    return outputTensor;
}

@end
