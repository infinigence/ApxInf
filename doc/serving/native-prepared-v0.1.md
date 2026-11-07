# Native MiniCPM5 prepared-code profile

This opt-in profile uses `apxinf-worker/2.7` and provider `native-minicpm5-prepared`.
It retains one model-owned prepared-code entry after request settlement.
Earlier native profiles retain their release-all behavior.
The profile supports one greedy sequence without sessions or prefix reuse.

## Identity and process policy

The runtime uses the pinned MLX 0.32.3 SDK and the checked native library.
The adapter revision is `apxinf-native-minicpm5-prepared-v1`.
Its execution identity extends the ordinary native identity with these fields:

| Field | Meaning |
| --- | --- |
| `preparation_policy` | Fixed string `single-code-entry-v1`. |
| `allocator_cache_limit_bytes` | The process-wide MLX allocator cache limit. |
| `retained_memory_limit_bytes` | Maximum active plus cached allocation after request cleanup. |

Both limits are positive safe integers and do not exceed `memory_limit_bytes`.
The launcher derives the retained limit from total budget minus sequence reservation.
The execution owner sets the cache limit before model loading.
Model families do not change process-wide allocator policy.
These allocator limits do not measure or guarantee process physical footprint.

## Preparation lifetime

The model retains only compiled code and immutable preparation metadata.
The entry key includes prompt length, output reservation, capacity, attention mode and prefill plan.
A changed key replaces the entry and its shape-dependent local functions.
The model does not retain request KV, taps or generated outputs.
The next request allocates fresh KV before execution.
Preparation or execution failure invalidates the retained entry.

## Settlement

Each `resources_released` event adds required `memory` with `active_bytes`, `cache_bytes` and `peak_bytes`.
All three values are non-negative safe integers.
The worker measures them after device synchronization and request cleanup.
The peak count must be at least the active count.
Released lease IDs retain the serial profile's meaning.
The retained lease array remains empty because prepared code belongs to the model instance.

The worker evicts its prepared entry if retained allocation exceeds the declared retained limit.
It then clears unused allocator cache and measures memory again.
Uncertain synchronization, release or accounting prevents successful settlement.
The coordinator updates resident accounting from the checked settlement record.
It checks resident allocation plus sequence reservation before another execution.
A public success still requires checked resource settlement.

The profile preserves the public HTTP request and response schemas.
Earlier protocols reject the new settlement field.
This document does not establish device qualification or a performance improvement.
