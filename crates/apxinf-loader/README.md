# Weight loading

`apxinf-loader::safetensors::NativeSafetensorsReader` opens an existing
SafeTensors file, sharded index, or checkpoint directory and materializes only
requested tensors. It preserves the same native dtype and path conventions as
`load_native_path`; BF16 is not upcast, and packed U8 uses `E2M1Pair`.

```rust,no_run
use std::path::Path;
use apxinf_loader::safetensors::NativeSafetensorsReader;

fn main() -> Result<(), String> {
    let mut reader = NativeSafetensorsReader::open_path(Path::new("/models/checkpoint"))?;
    if let Some(header) = reader.tensor_metadata("model.norm.weight") {
        println!("{:?} {:?}", header.dtype, header.shape);
    }
    assert_eq!(reader.payload_bytes_copied(), 0);
    let tensor = reader.load_tensor("model.norm.weight")?;
    // Upload/consume this owned CPU tensor, then drop it before loading another.
    drop(tensor);
    Ok(())
}
```

Opening validates header shapes/byte lengths and payload bounds without creating
owned tensor copies. Indexed checkpoints also validate shard paths and tensor
assignments. Header queries do not materialize payloads; returned tensors remain
valid after the reader is dropped. The copied-byte counter is cumulative across
successful loads, including repeated requests for the same name. It measures
owned payload copies, not process RSS, file-cache residency or GPU memory.
Checkpoint files must remain unchanged while their maps are held by a reader.

The existing eager loading APIs retain their behavior. This reader provides a
CPU loading primitive; device upload and model execution remain caller-owned.

Run deterministic CPU contracts:

```bash
cargo test --locked -p apxinf-loader
```

A separate read-only test can verify an existing real checkpoint in place:

```bash
APXINF_TEST_SAFETENSORS=/path/to/checkpoint \
  cargo test --locked -p apxinf-loader --test native_safetensors_reader \
  real_checkpoint_selected_payloads_match_independent_file_reads -- --ignored --nocapture
```

It checks headers, selects the smallest nonempty tensor of each dtype up to
1 MiB, and compares returned bytes with independent seek/read operations. It
prints checkpoint size, tensor count and copied bytes. It requires no GPU,
reference framework or checkpoint download. CI runs the synthetic contracts;
the real-checkpoint test is opt-in.
