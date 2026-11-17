use super::*;
use apxinf_model::vla::{Action, InferenceSpec, InitialLatent, PreparedInference, VlaRuntime};

struct EchoRuntime;

impl VlaRuntime for EchoRuntime {
    fn contract(&self) -> VlaContract {
        VlaContract {
            action_shape: [2, 2],
            patch_shape: [2, 3],
            max_token_len: 8,
            num_views: 1,
            image_size: 2,
            patch_size: 1,
            accepts_rgb_u8: false,
        }
    }

    fn infer(&self, _request: &VlaRequest<'_>) -> apxinf_core::Result<Action> {
        unreachable!()
    }

    fn prepare(&self, _spec: &InferenceSpec) -> apxinf_core::Result<Box<dyn PreparedInference>> {
        unreachable!()
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> apxinf_core::Result<Vec<f32>> {
        let VisionObservation::Patches(pixels) = &request.observation.vision else {
            panic!("expected patches");
        };
        let InitialLatent::Provided(noise) = request.initial_latent else {
            panic!("expected provided noise");
        };
        let state = request.observation.state.as_ref().unwrap();
        assert_eq!(pixels.dtype(), state.dtype());
        assert_eq!(pixels.dtype(), noise.dtype());
        assert_eq!(request.observation.token_ids, [1, 2]);
        assert_eq!(request.metadata.attention_mask.unwrap(), [1, 1]);
        assert_eq!(request.metadata.image_grid_thw.unwrap(), [[1, 1, 2]]);
        let mut output = noise.to_f32_vec()?;
        output[0] = pixels.to_f32_vec()?[0];
        output[1] = state.to_f32_vec()?[0];
        if let Some(planning) = request.metadata.planning {
            output[2] = planning.num_steps.unwrap_or(0) as f32;
            if let Some(reasoning) = &planning.reasoning {
                assert_eq!(reasoning.min_new_tokens, 1);
                assert_eq!(reasoning.terminator_ids, [7]);
                assert_eq!(reasoning.closing_ids, [8]);
                output[3] = reasoning.max_new_tokens as f32;
            }
        }
        if output[0] < 0.0 {
            return Err(apxinf_core::Error::Other("test runtime failure".into()));
        }
        Ok(output)
    }

    fn calibration_amax(
        &self,
        request: &VlaRequest<'_>,
    ) -> apxinf_core::Result<BTreeMap<String, f32>> {
        assert_eq!(request.metadata.embodiment_id, Some(2));
        let output = self.infer_host_f32(request)?;
        Ok(BTreeMap::from([("test.input".into(), output[0])]))
    }
}

fn run_python_test(code: &str) {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let runtime = EchoRuntime;
        let contract = runtime.contract();
        let runner = Py::new(
            py,
            ModelRunner {
                model: LoadedModel::Vla(Box::new(runtime)),
                contract,
                action_tokens: None,
                device: Device::Cpu,
                sampling_seed: Cell::new(0),
                sampling_draw: Cell::new(0),
            },
        )
        .unwrap();
        let locals = PyDict::new_bound(py);
        locals.set_item("runner", runner).unwrap();
        py.run_bound(
            r#"
import inspect
import numpy as np
pixels = np.full((2, 3), 1.0001, dtype=np.float32)
state = np.full((1, 1, 2), 2.0, dtype=np.float32)
noise = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
inputs = (pixels, np.array([[1, 1, 2]], dtype=np.uint32),
          np.array([1, 2], dtype=np.uint32), np.ones(2, dtype=np.uint8),
          state, 2, noise)
def expect_error(kind, message, callback):
    try:
        callback()
    except kind as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError("expected " + kind.__name__)
"#,
            Some(&locals),
            None,
        )
        .unwrap();
        py.run_bound(code, Some(&locals), None).unwrap();
    });
}

#[test]
fn legacy_preprocessed_keeps_f32_array_return_and_planning_options() {
    run_python_test(
        r#"
signature = inspect.signature(runner._infer_preprocessed)
assert "return_timing" not in signature.parameters
assert signature.parameters["num_steps"].kind == inspect.Parameter.KEYWORD_ONLY
assert signature.parameters["num_steps"].default is None
output = runner._infer_preprocessed(*inputs)
assert isinstance(output, np.ndarray) and output.dtype == np.float32
np.testing.assert_array_equal(output, [[pixels[0, 0], 2], [2, 3]])
output = runner._infer_preprocessed(*inputs, num_steps=5, max_new_tokens=6,
    min_new_tokens=1, terminator_ids=[7], closing_ids=[8])
np.testing.assert_array_equal(output, [[pixels[0, 0], 2], [5, 6]])
assert runner._calibrate_preprocessed(*inputs) == {"test.input": float(pixels[0, 0])}
assert runner.patches_per_view == 2
"#,
    );
}

#[test]
fn diagnostic_preprocessed_returns_timing_and_preserves_bf16_bits() {
    run_python_test(
        r#"
expected = runner._infer_preprocessed(*inputs)
output, model_ms = runner._infer_preprocessed_timed(*inputs)
np.testing.assert_array_equal(output, expected)
assert isinstance(model_ms, float) and np.isfinite(model_ms) and model_ms >= 0
bits = list(inputs)
for index in (0, 4, 6):
    bits[index] = (inputs[index].view(np.uint32) >> 16).astype(np.uint16)
output, model_ms = runner._infer_preprocessed_timed(*bits)
np.testing.assert_array_equal(output, [[1, 2], [2, 3]])
assert output.dtype == np.float32 and np.isfinite(model_ms) and model_ms >= 0
output, _ = runner._infer_preprocessed_timed(*bits, num_steps=5, max_new_tokens=6,
    min_new_tokens=1, terminator_ids=[7], closing_ids=[8])
np.testing.assert_array_equal(output, [[1, 2], [5, 6]])
expect_error(TypeError, "return_timing", lambda: runner._infer_preprocessed(*inputs, return_timing=True))
"#,
    );
}

#[test]
fn legacy_preprocessed_rejects_invalid_types_shapes_and_strides() {
    run_python_test(
        r#"
for method in (runner._infer_preprocessed, runner._calibrate_preprocessed):
    for index in (0, 4, 6):
        for dtype in (np.uint16, np.float64):
            invalid = list(inputs)
            invalid[index] = invalid[index].astype(dtype)
            expect_error(TypeError, "", lambda: method(*invalid))
    invalid = list(inputs)
    invalid[0] = pixels.reshape(-1)
    expect_error(TypeError, "", lambda: method(*invalid))
    for index, name in ((0, "pixel_values"), (4, "state"), (6, "noise")):
        invalid = list(inputs)
        invalid[index] = np.repeat(inputs[index], 2, axis=-1)[..., ::2]
        expect_error(ValueError, name + " must be C-contiguous float32", lambda: method(*invalid))
    invalid = list(inputs)
    invalid[3] = np.ones(1, dtype=np.uint8)
    expect_error(ValueError, "attention mask length", lambda: method(*invalid))
    invalid = list(inputs)
    invalid[6] = np.zeros((3, 2), dtype=np.float32)
    expect_error(ValueError, "noise expected", lambda: method(*invalid))
"#,
    );
}

#[test]
fn preprocessed_entries_preserve_request_and_runtime_errors() {
    run_python_test(
        r#"
for method in (runner._infer_preprocessed, runner._infer_preprocessed_timed):
    expect_error(ValueError, "reasoning requires terminator_ids", lambda: method(*inputs, max_new_tokens=2))
    expect_error(ValueError, "reasoning options require max_new_tokens", lambda: method(*inputs, min_new_tokens=1))
    invalid = list(inputs)
    invalid[0] = -pixels
    expect_error(RuntimeError, "test runtime failure", lambda: method(*invalid))
invalid = list(inputs)
invalid[0] = pixels.astype(np.float64)
expect_error(TypeError, "", lambda: runner._infer_preprocessed_timed(*invalid))
invalid[0] = np.repeat(pixels, 2, axis=-1).astype(np.uint16)[..., ::2]
expect_error(ValueError, "pixel_values must be C-contiguous BF16 uint16 bits",
             lambda: runner._infer_preprocessed_timed(*invalid))
"#,
    );
}
