// Core ML bridge for apxinf-coreml. Objective-C++ with ARC; C++ exceptions and Objective-C errors become -1
// and a thread-local message. No Python, no coremltools.
#include "bridge.h"
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <limits>

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 140400
API_AVAILABLE(macos(14.4))
@interface ApxCoreMLPlanRequest : NSObject
@property(nonatomic, strong) NSURL* url;
@property(nonatomic, strong) MLModelConfiguration* configuration;
@property(nonatomic, strong) MLComputePlan* result;
@property(nonatomic, strong) NSError* error;
@property(nonatomic, strong) NSString* completionError;
@property(nonatomic, strong) dispatch_semaphore_t done;
@end
@implementation ApxCoreMLPlanRequest
@end
#endif

namespace {
thread_local std::string last_error;

int fail(const std::string& message) {
    last_error = message;
    return -1;
}

int fail(NSError* error, const char* what) {
    return fail(std::string(what) + ": " + (error ? error.localizedDescription.UTF8String : "unknown error"));
}

template <typename Action> int guarded(Action action) {
    try {
        @try { return action(); }
        @catch (NSException* exception) {
            return fail(std::string("Core ML exception: ") + (exception.reason.UTF8String ?: "unknown"));
        }
    } catch (const std::exception& exception) {
        return fail(exception.what());
    } catch (...) {
        return fail("unknown native Core ML exception");
    }
}

MLModelConfiguration* configuration_for(int32_t units) {
    MLModelConfiguration* configuration = [MLModelConfiguration new];
    switch (units) {
        case 0: configuration.computeUnits = MLComputeUnitsAll; break;
        case 1: configuration.computeUnits = MLComputeUnitsCPUOnly; break;
        case 2: configuration.computeUnits = MLComputeUnitsCPUAndGPU; break;
        case 3: configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine; break;
        default: return nil;
    }
    return configuration;
}

ApxPlanString plan_string(NSString* value) {
    if (!value || !value.UTF8String) throw std::runtime_error("missing Core ML plan string");
    return {reinterpret_cast<const uint8_t*>(value.UTF8String),
            [value lengthOfBytesUsingEncoding:NSUTF8StringEncoding]};
}

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 140400
using PlanCompletion = void (^)(MLComputePlan*, NSError*);

PlanCompletion plan_completion(ApxCoreMLPlanRequest* request) API_AVAILABLE(macos(14.4)) {
    return ^(MLComputePlan* plan, NSError* failure) {
        try {
            @try { request.result = plan; request.error = failure; }
            @catch (NSException* exception) { request.completionError = exception.reason ?: @"plan completion exception"; }
        } catch (...) { request.completionError = @"native plan completion exception"; }
        dispatch_semaphore_signal(request.done);
    };
}

ApxPlanDevice plan_device(id<MLComputeDeviceProtocol> device) API_AVAILABLE(macos(14.4)) {
    int32_t kind = 0;
    if ([(id)device isKindOfClass:MLCPUComputeDevice.class]) kind = 1;
    else if ([(id)device isKindOfClass:MLGPUComputeDevice.class]) kind = 2;
    else if ([(id)device isKindOfClass:MLNeuralEngineComputeDevice.class]) kind = 3;
    return {kind, plan_string(NSStringFromClass([(id)device class]))};
}

void walk_plan(MLComputePlan* plan, MLModelStructureProgramBlock* block, size_t function_index,
               std::vector<size_t> path, size_t& row_count, ApxPlanVisitor visitor, void* context)
               API_AVAILABLE(macos(14.4)) {
    if (path.size() > 128) throw std::runtime_error("Core ML plan nesting limit exceeded");
    size_t index = 0;
    for (MLModelStructureProgramOperation* operation in block.operations) {
        if (++row_count > 100000) throw std::runtime_error("Core ML plan operation limit exceeded");
        std::vector<size_t> operation_path = path;
        operation_path.push_back(index++);
        std::vector<std::vector<ApxPlanBinding>> bindings;
        std::vector<ApxPlanInput> inputs;
        bindings.reserve(operation.inputs.count);
        inputs.reserve(operation.inputs.count);
        for (NSString* name in [operation.inputs.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
            MLModelStructureProgramArgument* argument = operation.inputs[name];
            std::vector<ApxPlanBinding> group;
            for (MLModelStructureProgramBinding* binding in argument.bindings) {
                group.push_back({binding.name != nil, binding.name ? plan_string(binding.name) : ApxPlanString{},
                                 binding.value != nil});
            }
            bindings.push_back(std::move(group));
            inputs.push_back({plan_string(name), bindings.back().data(), bindings.back().size()});
        }
        std::vector<ApxPlanString> outputs;
        for (MLModelStructureProgramNamedValueType* output in operation.outputs)
            outputs.push_back(plan_string(output.name));
        std::vector<size_t> children;
        for (MLModelStructureProgramBlock* child in operation.blocks) children.push_back(child.operations.count);
        MLComputePlanDeviceUsage* usage = [plan computeDeviceUsageForMLProgramOperation:operation];
        std::vector<ApxPlanDevice> supported;
        for (id<MLComputeDeviceProtocol> device in usage.supportedComputeDevices)
            supported.push_back(plan_device(device));
        const auto preferred = usage.preferredComputeDevice;
        ApxPlanRow row{function_index, operation_path.data(), operation_path.size(), plan_string(operation.operatorName),
                       inputs.data(), inputs.size(), outputs.data(), outputs.size(), children.data(), children.size(),
                       usage != nil, supported.data(), supported.size(), preferred != nil,
                       preferred ? plan_device(preferred) : ApxPlanDevice{}};
        if (visitor(context, nullptr, &row) != 0) throw std::runtime_error("Core ML plan visitor rejected a row");
        size_t child_index = 0;
        for (MLModelStructureProgramBlock* child in operation.blocks) {
            auto child_path = operation_path;
            child_path.push_back(child_index++);
            walk_plan(plan, child, function_index, std::move(child_path), row_count, visitor, context);
        }
    }
}
#endif

NSArray<NSNumber*>* shape_of(const int64_t* shape, size_t rank) {
    NSMutableArray<NSNumber*>* out = [NSMutableArray arrayWithCapacity:rank];
    for (size_t i = 0; i < rank; ++i) [out addObject:@(shape[i])];
    return out;
}

NSArray<NSNumber*>* contiguous_strides(const int64_t* shape, size_t rank) {
    std::vector<int64_t> strides(rank, 1);
    for (size_t i = rank; i-- > 1;) strides[i - 1] = strides[i] * shape[i];
    NSMutableArray<NSNumber*>* out = [NSMutableArray arrayWithCapacity:rank];
    for (size_t i = 0; i < rank; ++i) [out addObject:@(strides[i])];
    return out;
}

// Wraps the caller's FP16 buffers; the arrays borrow them for the duration of one bridge call only.
MLDictionaryFeatureProvider* features_of(const char* const* names, const uint16_t* const* inputs,
                                         const int64_t* const* shapes, const size_t* ranks, size_t count) {
    NSMutableDictionary<NSString*, MLFeatureValue*>* features = [NSMutableDictionary dictionary];
    for (size_t i = 0; i < count; ++i) {
        NSError* error = nil;
        MLMultiArray* array = [[MLMultiArray alloc]
            initWithDataPointer:(void*)inputs[i]
                          shape:shape_of(shapes[i], ranks[i])
                       dataType:MLMultiArrayDataTypeFloat16
                        strides:contiguous_strides(shapes[i], ranks[i])
                    deallocator:nil
                          error:&error];
        if (!array) {
            fail(error, "cannot wrap a Core ML input");
            return nil;
        }
        features[@(names[i])] = [MLFeatureValue featureValueWithMultiArray:array];
    }
    NSError* error = nil;
    MLDictionaryFeatureProvider* provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
    if (!provider) fail(error, "cannot build Core ML inputs");
    return provider;
}

// Copies each named FP16 output into a contiguous buffer, honouring the strides Core ML reports.
int copy_outputs(id<MLFeatureProvider> result, const char* const* output_names, uint16_t* const* outputs,
                 const int64_t* const* output_shapes, const size_t* output_ranks, size_t output_count) {
    for (size_t o = 0; o < output_count; ++o) {
        MLMultiArray* array = [result featureValueForName:@(output_names[o])].multiArrayValue;
        if (!array || array.dataType != MLMultiArrayDataTypeFloat16 || array.shape.count != output_ranks[o])
            return fail(std::string("Core ML output is not the expected FP16 array: ") + output_names[o]);
        const size_t rank = output_ranks[o];
        size_t count = 1;
        for (size_t d = 0; d < rank; ++d) {
            if (array.shape[d].longLongValue != output_shapes[o][d])
                return fail(std::string("Core ML output shape differs: ") + output_names[o]);
            count *= (size_t)output_shapes[o][d];
        }
        std::vector<int64_t> strides(rank);
        for (size_t d = 0; d < rank; ++d) strides[d] = array.strides[d].longLongValue;
        // Rows of the innermost dimension are copied whole when it is unit-stride (the usual case,
        // with or without padding between rows); other layouts are copied element by element.
        const int64_t row = rank ? output_shapes[o][rank - 1] : 1;
        const bool rows_contiguous = rank == 0 || strides[rank - 1] == 1;
        // Dimensions advanced by the index counter: all but the innermost when whole rows are copied.
        const size_t outer = rows_contiguous && rank ? rank - 1 : rank;
        __block int status = 0;
        [array getBytesWithHandler:^(const void* bytes, NSInteger size) {
            const uint16_t* src = (const uint16_t*)bytes;
            std::vector<int64_t> index(rank, 0);
            const size_t step = rows_contiguous ? (size_t)row : 1;
            for (size_t n = 0; n < count; n += step) {
                int64_t at = 0;
                for (size_t d = 0; d < rank; ++d) at += index[d] * strides[d];
                if (at < 0 || (at + (int64_t)step) * 2 > size) { status = -1; return; }
                if (rows_contiguous) {
                    memcpy(outputs[o] + n, src + at, step * 2);
                } else {
                    outputs[o][n] = src[at];
                }
                for (size_t d = outer; d-- > 0;) {
                    if (++index[d] < output_shapes[o][d]) break;
                    index[d] = 0;
                }
            }
        }];
        if (status) return fail(std::string("Core ML output strides exceed its buffer: ") + output_names[o]);
    }
    return 0;
}
}  // namespace

extern "C" {
const char* apx_coreml_error(void) { return last_error.c_str(); }

int apx_coreml_compile(const char* package, const char* compiled) {
    return guarded([&]() -> int {
    @autoreleasepool {
        NSURL* source = [NSURL fileURLWithPath:@(package)];
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        __block NSURL* result = nil;
        __block NSError* error = nil;
        [MLModel compileModelAtURL:source completionHandler:^(NSURL* url, NSError* e) {
            result = url;
            error = e;
            dispatch_semaphore_signal(done);
        }];
        dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
        if (!result) return fail(error, "Core ML compilation failed");
        NSError* move = nil;
        if (![[NSFileManager defaultManager] moveItemAtURL:result toURL:[NSURL fileURLWithPath:@(compiled)] error:&move])
            return fail(move, "cannot move the compiled Core ML model");
        return 0;
    }
    });
}

int apx_coreml_load(const char* compiled, int32_t compute_units, void** out) {
    return guarded([&]() -> int {
    @autoreleasepool {
        MLModelConfiguration* config = configuration_for(compute_units);
        if (!config) return fail("unknown Core ML compute units");
        NSError* error = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(compiled)] configuration:config error:&error];
        if (!model) return fail(error, "cannot load the compiled Core ML model");
        *out = (__bridge_retained void*)model;
        return 0;
    }
    });
}

int apx_coreml_inspect_plan(const char* compiled, int32_t units, uint64_t timeout_ms,
                           ApxPlanVisitor visitor, void* context) {
    return guarded([&]() -> int {
        @autoreleasepool {
            if (!compiled || !visitor || !context || timeout_ms == 0 ||
                timeout_ms > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() / NSEC_PER_MSEC))
                return fail("invalid Core ML plan inspection arguments");
            MLModelConfiguration* configuration = configuration_for(units);
            if (!configuration) return fail("unknown Core ML compute units");
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 140400
            if (@available(macOS 14.4, *)) {
                ApxCoreMLPlanRequest* request = [ApxCoreMLPlanRequest new];
                request.url = [NSURL fileURLWithPath:@(compiled)];
                request.configuration = configuration;
                request.done = dispatch_semaphore_create(0);
                [MLComputePlan loadContentsOfURL:request.url configuration:request.configuration
                    completionHandler:plan_completion(request)];
                if (dispatch_semaphore_wait(request.done, dispatch_time(DISPATCH_TIME_NOW,
                        static_cast<int64_t>(timeout_ms) * NSEC_PER_MSEC)) != 0)
                    return fail("Core ML plan wait timed out; framework work is not cancelled");
                if (request.completionError) return fail(request.completionError.UTF8String);
                MLComputePlan* result = request.result;
                if (request.error || !result) return fail(request.error, "cannot inspect the Core ML compute plan");
                auto program = result.modelStructure.program;
                if (!program) return fail("Core ML placement inspection supports ML Programs only");
                size_t function_index = 0;
                size_t row_count = 0;
                for (NSString* name in [program.functions.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
                    auto block = program.functions[name].block;
                    if (!block) return fail("Core ML plan function has no block");
                    ApxPlanFunction function{plan_string(name), block.operations.count};
                    if (visitor(context, &function, nullptr) != 0) return fail("Core ML plan visitor rejected a function");
                    walk_plan(result, block, function_index++, {}, row_count, visitor, context);
                }
                return 0;
            }
#endif
            return fail("Core ML compute-plan inspection requires macOS and SDK 14.4 or newer");
        }
    });
}

void apx_coreml_free(void* model) {
    if (model) CFRelease(model);
}

int apx_coreml_predict(void* handle, const char* const* input_names, const uint16_t* const* inputs,
                       const int64_t* const* input_shapes, const size_t* input_ranks, size_t input_count,
                       const char* const* output_names, uint16_t* const* outputs, const int64_t* const* output_shapes,
                       const size_t* output_ranks, size_t output_count) {
    return guarded([&]() -> int {
    @autoreleasepool {
        MLModel* model = (__bridge MLModel*)handle;
        MLDictionaryFeatureProvider* provider = features_of(input_names, inputs, input_shapes, input_ranks, input_count);
        if (!provider) return -1;
        NSError* error = nil;
        id<MLFeatureProvider> result = [model predictionFromFeatures:provider error:&error];
        if (!result) return fail(error, "Core ML prediction failed");
        return copy_outputs(result, output_names, outputs, output_shapes, output_ranks, output_count);
    }
}

int apx_coreml_predict_batch(void* handle, size_t batch, const char* const* input_names, const uint16_t* const* inputs,
                             const int64_t* const* input_shapes, const size_t* input_ranks, size_t input_count,
                             const char* const* output_names, uint16_t* const* outputs,
                             const int64_t* const* output_shapes, const size_t* output_ranks, size_t output_count) {
    @autoreleasepool {
        MLModel* model = (__bridge MLModel*)handle;
        NSMutableArray<id<MLFeatureProvider>>* items = [NSMutableArray arrayWithCapacity:batch];
        for (size_t b = 0; b < batch; ++b) {
            MLDictionaryFeatureProvider* provider =
                features_of(input_names, inputs + b * input_count, input_shapes, input_ranks, input_count);
            if (!provider) return -1;
            [items addObject:provider];
        }
        NSError* error = nil;
        id<MLBatchProvider> results =
            [model predictionsFromBatch:[[MLArrayBatchProvider alloc] initWithFeatureProviderArray:items] error:&error];
        if (!results) return fail(error, "Core ML batch prediction failed");
        if ((size_t)results.count != batch) return fail("Core ML batch prediction returned another item count");
        for (size_t b = 0; b < batch; ++b) {
            int status = copy_outputs([results featuresAtIndex:(NSInteger)b], output_names, outputs + b * output_count,
                                      output_shapes, output_ranks, output_count);
            if (status) return status;
        }
        return 0;
    }
    });
}
}
