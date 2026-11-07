#include <atomic>
#include <chrono>
#include <thread>
#include "../native/bridge.mm"

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 140400
static std::atomic<int> released{0};

API_AVAILABLE(macos(14.4))
@interface PlanReleaseProbe : ApxCoreMLPlanRequest
@end
@implementation PlanReleaseProbe
- (void)dealloc { ++released; }
@end

API_AVAILABLE(macos(14.4))
@interface ThrowingPlanProbe : PlanReleaseProbe
@end
@implementation ThrowingPlanProbe
- (void)setResult:(MLComputePlan*)value {
    [NSException raise:@"PlanProbeException" format:@"injected completion exception"];
}
@end
#endif

int main() {
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 140400
    if (@available(macOS 14.4, *)) {
        const MLComputeUnits expected[] = {MLComputeUnitsAll, MLComputeUnitsCPUOnly,
                                           MLComputeUnitsCPUAndGPU, MLComputeUnitsCPUAndNeuralEngine};
        for (int32_t index = 0; index < 4; ++index)
            if (configuration_for(index).computeUnits != expected[index]) return 1;
        if (configuration_for(-1) || configuration_for(4)) return 2;
        dispatch_semaphore_t release_gate = dispatch_semaphore_create(0);
        dispatch_semaphore_t completed = dispatch_semaphore_create(0);
        __weak PlanReleaseProbe* weak_request = nil;
        @autoreleasepool {
            PlanReleaseProbe* request = [PlanReleaseProbe new];
            weak_request = request;
            request.url = [[NSURL alloc] initFileURLWithPath:@"/tmp/apxinf-plan-lifetime-only"];
            request.configuration = configuration_for(1);
            request.done = dispatch_semaphore_create(0);
            PlanCompletion completion = plan_completion(request);
            dispatch_async(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
                dispatch_semaphore_wait(release_gate, DISPATCH_TIME_FOREVER);
                completion(nil, nil);
                dispatch_semaphore_signal(completed);
            });
            if (dispatch_semaphore_wait(request.done, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_MSEC)) == 0)
                return 3;
        }
        if (!weak_request || released.load() != 0 || !weak_request.url || !weak_request.configuration) return 4;
        dispatch_semaphore_signal(release_gate);
        if (dispatch_semaphore_wait(completed, dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC)) != 0) return 5;
        for (int attempt = 0; released.load() == 0 && attempt < 200; ++attempt)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (weak_request || released.load() != 1) return 6;
        @autoreleasepool {
            ThrowingPlanProbe* request = [ThrowingPlanProbe new];
            request.done = dispatch_semaphore_create(0);
            plan_completion(request)(nil, nil);
            if (dispatch_semaphore_wait(request.done, DISPATCH_TIME_NOW) != 0) return 7;
            if (![request.completionError isEqualToString:@"injected completion exception"]) return 8;
        }
        @autoreleasepool {
            PlanReleaseProbe* request = [PlanReleaseProbe new];
            request.done = dispatch_semaphore_create(0);
            NSError* error = [NSError errorWithDomain:@"PlanProbe" code:1 userInfo:nil];
            plan_completion(request)(nil, error);
            if (dispatch_semaphore_wait(request.done, DISPATCH_TIME_NOW) != 0 || request.result || request.error != error)
                return 9;
        }
        return released.load() == 3 ? 0 : 10;
    }
#endif
    return 77;
}
