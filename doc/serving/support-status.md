# Serving support and verification boundaries

The [interface contract](contracts-v0.1.md) defines serving semantics.
The [serial profile](serial-profile-v0.1.md) defines the implemented protocol subset.
- [Serial service lifecycle and public HTTP interface](local-service-v0.1.md) defines its own supported scope and limits.

Provider profiles are explicit selections. External inference providers do not establish native ApxInf model support.
The system design contains future stages; sessions and continuous batching are not implied by serial conformance.

## Verification scope

CPU contract and lifecycle tests do not establish device computation, model quality, or performance.
Historical observations span different source and runtime revisions and cannot form one qualification matrix.
This submission cleanup does not rerun or extend those observations.
Use the declared model, runtime, source, inputs and generation policy for each verification.
Keep complete outputs, failed cases and cleanup evidence with the run.
Successful transport and resource settlement do not establish task completion.
Allocator counters, process RSS and system swap describe different quantities.

Raw outputs, environment snapshots and experiment histories remain in ignored development records.
Those local records are not portable files in a fresh checkout.
The [serving PR](https://github.com/infinigence/ApxInf/pull/116) carries the implementation scope and remaining upstream review.
