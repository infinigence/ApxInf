// Core ML bridge for apxinf-coreml. Objective-C++ with ARC; C++ exceptions and Objective-C errors become -1
// and a thread-local message. No Python, no coremltools.
#include "bridge.h"
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;

int fail(const std::string& message) {
    last_error = message;
    return -1;
}

int fail(NSError* error, const char* what) {
    return fail(std::string(what) + ": " + (error ? error.localizedDescription.UTF8String : "unknown error"));
}

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
}  // namespace

extern "C" {
const char* apx_coreml_error(void) { return last_error.c_str(); }

int apx_coreml_compile(const char* package, const char* compiled) {
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
}

int apx_coreml_load(const char* compiled, int32_t compute_units, void** out) {
    @autoreleasepool {
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        switch (compute_units) {
            case 0: config.computeUnits = MLComputeUnitsAll; break;
            case 1: config.computeUnits = MLComputeUnitsCPUOnly; break;
            case 2: config.computeUnits = MLComputeUnitsCPUAndGPU; break;
            case 3: config.computeUnits = MLComputeUnitsCPUAndNeuralEngine; break;
            default: return fail("unknown Core ML compute units");
        }
        NSError* error = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(compiled)] configuration:config error:&error];
        if (!model) return fail(error, "cannot load the compiled Core ML model");
        *out = (__bridge_retained void*)model;
        return 0;
    }
}

void apx_coreml_free(void* model) {
    if (model) CFRelease(model);
}

int apx_coreml_predict(void* handle, const char* const* input_names, const uint16_t* const* inputs,
                       const int64_t* const* input_shapes, const size_t* input_ranks, size_t input_count,
                       const char* const* output_names, uint16_t* const* outputs, const int64_t* const* output_shapes,
                       const size_t* output_ranks, size_t output_count) {
    @autoreleasepool {
        MLModel* model = (__bridge MLModel*)handle;
        NSMutableDictionary<NSString*, MLFeatureValue*>* features = [NSMutableDictionary dictionary];
        for (size_t i = 0; i < input_count; ++i) {
            NSError* error = nil;
            // The array borrows the caller's buffer for the duration of this call only.
            MLMultiArray* array = [[MLMultiArray alloc]
                initWithDataPointer:(void*)inputs[i]
                              shape:shape_of(input_shapes[i], input_ranks[i])
                           dataType:MLMultiArrayDataTypeFloat16
                            strides:contiguous_strides(input_shapes[i], input_ranks[i])
                        deallocator:nil
                              error:&error];
            if (!array) return fail(error, "cannot wrap a Core ML input");
            features[@(input_names[i])] = [MLFeatureValue featureValueWithMultiArray:array];
        }
        NSError* error = nil;
        MLDictionaryFeatureProvider* provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
        if (!provider) return fail(error, "cannot build Core ML inputs");
        id<MLFeatureProvider> result = [model predictionFromFeatures:provider error:&error];
        if (!result) return fail(error, "Core ML prediction failed");
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
            __block int status = 0;
            [array getBytesWithHandler:^(const void* bytes, NSInteger size) {
                const uint16_t* src = (const uint16_t*)bytes;
                std::vector<int64_t> index(rank, 0);
                for (size_t n = 0; n < count; ++n) {
                    int64_t at = 0;
                    for (size_t d = 0; d < rank; ++d) at += index[d] * strides[d];
                    if (at < 0 || (at + 1) * 2 > size) { status = -1; return; }
                    outputs[o][n] = src[at];
                    for (size_t d = rank; d-- > 0;) {
                        if (++index[d] < output_shapes[o][d]) break;
                        index[d] = 0;
                    }
                }
            }];
            if (status) return fail(std::string("Core ML output strides exceed its buffer: ") + output_names[o]);
        }
        return 0;
    }
}
}
