# MiniCPM vision candidate observation

Date: 2026-10-23.
Product source revision: `1fc6a22`.
This observation examines a separate vision candidate through public dependency APIs.
It does not change the active text provider or qualify the original Claude drawing task.

## Candidate and runtime identity

The candidate is `openbmb/MiniCPM-V-4.7-1B`, revision `379f120c2ff3588e6e06559ec6b9ec15cc05114e`.
It differs from the existing MiniCPM5-2B text checkpoint.
The [model card](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B) declares image understanding and tool calling.
Those published capabilities do not establish ApxInf compatibility.

The isolated tool environment contains MLX-VLM 0.7.10, MLX 0.32.4, Transformers 5.18.0, Torch 2.12.1, and Torchvision 0.27.1.
MLX-VLM supplies model execution. Transformers supplies processor and tokenizer APIs.
Torch and Torchvision support processor construction. NumPy supplies host arrays, and Pillow decodes PNG files.
The resolved package record passes its dependency check.
The shared text environment and production dependency files remain unchanged.

The loaded model reports 1,554,476,272 parameters, all with BF16 dtype.
Its checkpoint weight file contains 3109054480 bytes.
Its SHA-256 is `8c4921ac83182c85190ad10c3aeb118d51e385d2d98ec13b88b514bbf6cac002`.
Both child manifests record all eight checkpoint artifacts.
The source revision and artifact hashes remain separate from runtime qualification.

The probe checks its source, Python executable, module paths, and actual MLX library before loading weights.
It checks resolved module and library paths against the isolated environment.
The loaded `libmlx.dylib` SHA-256 is `d8c9393d584e781393c3effccfccd8ba85ad0c70264edaf9f897dca9875375c6`.
The existing native text library retains its separate identity.

## Authoritative image preparation

Model discovery changes which processor the dependency selects.
A standalone processor load initially selects the Transformers processor.
Importing the registered model module selects MLX-VLM's `MiniCPMV4_7Processor`, as the actual model load does.
A weight-free public discovery call establishes that preparation route before inference.
The runtime probe uses the processor returned by the actual strict model load.

The direct processor call requests NumPy arrays, one image slice, 16x visual downsampling, and disabled thinking.
It returns nested image arrays, `tgt_sizes`, image bounds, and explicit downsampling metadata.
The probe preserves this public output structure when it calls the public generation API.
It checks one actual image slice and 64 image placeholders before device transfer.
Earlier standalone processor observations do not qualify this selected model route.

## Two observed images

Both inputs are 960 by 960 PNG files.
The first is the retained failed pelican rendering from the [client transport observation](minicpm-vision-transport-validation-20261023.md).
The second is an original control image with a red square on a white background.
An original handwritten fixture script creates that control.
The model receives the same question about visible objects and whether a recognizable pelican rides a bicycle.

Both preparations contain the same 121 token IDs, including 64 image placeholders.
Both use the same checkpoint, source, library, processor options, and generation controls.
Each run uses a fresh process.
The image pixels differ.

| Observation | Failed pelican rendering | Original control image |
| --- | --- | --- |
| Yielded output IDs, including EOS | 105 | 54 |
| Final EOS ID | 248046 | 248046 |
| Described visible content | Green oval, rectangular blocks, and a triangular shape | Red square on a white background |
| Recognizable pelican or bicycle reported | No | No |
| Child result in parent process record | Zero, reaped, no signals | Zero, reaped, no signals |

Direct inspection of the failed rendering agrees with the description's main shapes and its negative pelican verdict.
The candidate also distinguishes the original control's color and shape.
These observations support image grounding for these two cases.
They do not establish general image understanding or reliable visual review.
Neither observation generates or revises an SVG.

The requested sampling controls are temperature 0.7, top-p 0.95, top-k 100, min-p zero, and seed 42.
The repetition penalty is 1.03 with an explicit context size of 8192.
Presence and frequency penalties are zero.
The generation API permits at most 512 yielded IDs, and the probe stops after a yielded EOS.
The dependency can sample ahead of its yielded output.
The records therefore count yielded IDs rather than claim an exact count of internal selections.

## Resource and lifecycle observations

The host reports normal pressure before each run on the 16 GiB Apple M4 system.
Each probe sets an 8 GiB MLX memory limit and a 256 MiB allocator cache limit.
Observed allocator peaks are 3378466126 and 3486174542 bytes.
These values describe the two measured cases, rather than a general capacity bound.

Both probes close the generator and synchronize before removing owned references.
Both report zero cleanup errors, 1056 active allocator bytes, and zero cached bytes after their cleanup sequence.
These values do not establish complete allocator release.
The parent separately records child exit zero and successful reaping without signals.
The root observes the outer wrapper retain the measurement lock through child settlement.
The two runs do not overlap.

## Remaining P2c and task gates

The [P2c plan](implementation-plan.md#p2c-serial-vision-requests) remains incomplete.
The current ApxInf gateway and worker protocols still reject image input.
A production vision profile needs the owned media contract, paired validators, original fixtures, and explicit provider selection.
It also needs reference processing and output comparisons, supported-shape limits, context checks, cancellation, and repeated requests in one worker.
These fresh-process observations do not establish those behaviors.

The original real Claude task still requires tool execution, a recognizable pelican riding a bicycle, and inspection of its actual artifact.
This candidate has not passed that task or its image-driven revision loop.
Vision batching and state reuse remain disabled pending their own checks.

## Review and retained records

The first independent source review identifies an incorrect processor assumption, late runtime checks, and incomplete failure handling.
The corrected probe uses the selected processor's observed structure and checks provenance before model loading.
It preserves primary errors and records cleanup errors and partial yielded counts separately.
A fresh independent source review accepts those corrections before either runtime observation.

Ignored records reside under `devlocal/serving-minicpm/vision-grounding/`.
The `vision-one-image/` and `vision-control-image/` directories retain captures, numeric IDs, output text, and independent parent process records.
Case files bind source and Python hashes. Separate launch records retain commands and host observations.
The directory also retains dependency versions, model metadata, processor observations, and source reviews.
These local records do not accompany a fresh Git checkout.

Structural writing checks do not establish a full dictionary review or certification.
