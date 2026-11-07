# apxinf_ref

Independent reference implementations that produce the numerical gold an
ApxInf accelerator port is verified against. Nothing here is linked into the
engine; the engine consumes the artifacts this code publishes.

One directory per model:

```
<model>/
  generate_<model>_torch_gold.py   entry point
  <package>/                        torch-free artifact side + torch capture side
  _vendor/                          vendored upstream snapshot, unmodified
  tests/
```

`pi05/` is the only model at present. Add a model by adding a sibling
directory; keep model directories self-contained rather than importing across
them. Do not name a directory `models/`: the repository `.gitignore` ignores
that name at every depth, so `git add` would silently skip the package.

## Install

`torch`/`torch_musa` come from the host's own wheels; the rest is pip-installable:

```bash
pip install numpy 'transformers==4.53.2' safetensors torchada
```

`torchada` is only needed for `--device musa`; the CLI imports it before torch,
so it must be installed in the environment that runs the entry point.

## Artifacts

Generated artifacts are never committed. They live under the ignored
`<repo-root>/devlocal/<model>-gold/<device>/`, one directory per host, for
example `devlocal/pi05-gold/orin/` and (later) `devlocal/pi05-gold/musa/`.
An artifact is the triple `inputs.npz` + `debug.npz` + `manifest.json`; keep
them together, since the manifest is the artifact's identity.