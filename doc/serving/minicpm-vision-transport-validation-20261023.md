# Claude rendered image transport

Date: 2026-10-23.
Product source revision: `a4f739a`.
This record qualifies one client transport observation, rather than model vision or task completion.
The original pelican drawing remains visually unaccepted.

## Question and boundaries

The current MiniCPM5-2B profile accepts text only.
The user asks why the model cannot inspect its rendered drawing and revise it.
That workflow requires client image transport, a public image interface, and a qualified vision model.
This observation tests the first requirement with the real client.

The existing [P2c plan](implementation-plan.md#p2c-serial-vision-requests) defines a separate serial vision branch after P1a.
The [prepared input contract](contracts-v0.1.md#prepared-input) requires owned media descriptors before implementation.
Those descriptors bind content identity, format, shape, position, byte count, and lifetime.
The vision adapter also defines effective context length.

## Observed client request

The probe runs Claude Code 2.1.289 with an isolated configuration and the bare, restricted profile.
It requests only `Read` and explicitly grants access to the owned PNG directory.
The client advertises `Read`, and the capture verifies the requested tool inventory.
The source PNG renders the previously failed bounded-profile SVG.
Its size is 38005 bytes.

A local server supplies synthetic Messages events that request one image Read, followed by a final text marker.
No model executes in this probe.
The real client executes the tool and sends two Messages requests.
Its second request contains a user `tool_result` for the supplied tool-use identifier.
That result contains one image block with a base64 source and media type `image/png`.

The decoded image contains exactly 38005 bytes.
Its SHA-256 is `ffea3dea84b395b31844f2cfb3524a5c55a38a0d1b230eb75a1eceb4b3ab4889`.
That digest equals the source PNG digest.
The client exits zero without a timeout or capture diagnostics.
The synthetic final marker does not establish image understanding.

## Current serving gap

The [gateway](../../crates/apxinf-serving/src/gateway.rs) rejects direct image blocks and images nested in tool results.
The nested image reaches `content_text`, which reports HTTP 400 `unsupported_feature`.
The public capability response correctly reports `images: false`.
No image reaches the current worker preparation path.

Both worker validators require string message content and token-only prepared input.
The native MiniCPM5 worker and Python MLX-LM worker advertise `image: false`.
The existing native Qwen3-VL model input accepts processed tensors below serving.
That internal model interface does not supply HTTP media transport or qualify MiniCPM vision execution.

This observation proves that the tested client can deliver this PNG through `Read`.
It identifies missing serving interfaces without attributing vision capability to the text checkpoint.
A separate vision profile requires a declared model, processor, runtime, precision, supported shapes, and media limits.
Its contract must preserve tool-result association and image order.
Its qualification must check effective context, processing, cancellation, memory, repeated requests, and actual output.
The existing text protocols and their declared capabilities retain their behavior.

## Review and retained evidence

Independent source readers trace the gateway, both validators, workers, supervisor, and P2c dependencies.
An independent synthesis reviewer checks the actual client request and decodes its PNG.
That reviewer checks the source digest, tool inventory, exit result, and limited interpretation.
The private probe extends the existing client observation with an optional image argument.
It reuses the maintained capture helper for process termination and tool-inventory checks.
The reviewer finds no substantive defect and no source comments in that extension.

The local records reside under ignored `devlocal/serving-minicpm/`.
`client-context-probe-0225f23c/image-read/` retains requests, capture events, source identity, and command arguments.
`vision-grounding/` retains the gateway and worker traces, synthesized flow, and separate external reference notes.
These records do not accompany a fresh Git checkout.
No production dependency, provider, interface, or capability changes during this observation.

Structural writing checks do not establish a full dictionary review or certification.
