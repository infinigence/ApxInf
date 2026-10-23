from __future__ import annotations

import gc
from contextlib import redirect_stdout
from pathlib import Path
import os
import threading
import time
import sys
from typing import Any

if __package__:
    from . import contracts
    from . import clef_contracts as schema
    from . import clef_identity as identity
else:
    import contracts
    import clef_contracts as schema
    import clef_identity as identity

ClefError = identity.ClefError


def encoding_document(encoded: Any) -> dict:
    if encoded.media is not None:
        raise ClefError("protocol_fault", "The text decision encoding unexpectedly contains media.")
    return {"input_ids": list(encoded.input_ids), "questions": [
        {"question_id": question.question_id, "question_type": question.question_type,
         "question_span": list(question.question_span), "option_spans": [list(span) for span in question.option_spans],
         "option_ids": list(question.option_ids)} for question in encoded.questions]}


class ClefRuntime:
    def __init__(self, model_path: str, loader_path: str, max_context: int, memory_limit_bytes: int):
        contracts._int(max_context, 1, 4096)
        contracts._int(memory_limit_bytes, 1)
        self.owner = threading.get_ident()
        self.max_context = max_context
        self.model_path = str(Path(model_path).absolute())
        root = Path(model_path)
        loader = Path(loader_path)
        artifacts, self.vocab_size = identity.model_artifacts(root)
        if artifacts != schema.approved_artifacts():
            raise ClefError("model_unavailable", "The Clef artifact digests differ from the approved checkpoint.")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        runtime, modules = identity.check_runtime()
        self.mx = modules["mlx_core"]
        self.mx.set_memory_limit(memory_limit_bytes)
        self.mx.set_cache_limit(0)
        with redirect_stdout(sys.stderr):
            self.dependency = identity.load_dependency(loader)
            self.model = self.dependency.load(str(root), head_dtype=self.mx.bfloat16, backend="lm")
        if self.model.processor is not None:
            raise ClefError("model_unavailable", "The Clef text profile requires a text-only processor.")
        self.mx.synchronize()
        checked_artifacts, checked_vocab = identity.model_artifacts(root)
        if checked_artifacts != artifacts or checked_vocab != self.vocab_size:
            raise ClefError("model_unavailable", "The Clef artifacts changed during loading.")
        if identity.file_digest(loader) != schema.LOADER_SHA256:
            raise ClefError("model_unavailable", "The Clef dependency changed during loading.")
        self.manifest = identity.manifest(loader, artifacts, runtime, identity.runtime_files(modules),
                                          max_context, memory_limit_bytes)
        self.capabilities = schema.capabilities(self.vocab_size, max_context)
        self.model_revision = contracts.canonical_identity_digest("apxinf-model-identity-v1", self.manifest)
        self.capability_revision = contracts.canonical_identity_digest("apxinf-capability-v1", self.capabilities)
        self.used = False

    def _owned(self) -> None:
        if threading.get_ident() != self.owner:
            raise ClefError("internal_error", "The Clef runtime was called outside its execution owner.")

    def memory(self) -> dict:
        self._owned()
        return {"active_bytes": self.mx.get_active_memory(), "peak_bytes": self.mx.get_peak_memory(),
                "cache_bytes": self.mx.get_cache_memory()}

    def encode(self, record: schema.Record) -> schema.Encoding:
        self._owned()
        try:
            with redirect_stdout(sys.stderr):
                encoded = self.dependency.encode_record(self.model.tokenizer, record.document(),
                    max_length=self.max_context, max_state_tokens=None, processor=None, truncate=False)
        except self.dependency.ContextTooLong as error:
            raise ClefError("context_limit", "The complete decision record exceeds the context limit.") from error
        return schema.validate_encoding(encoding_document(encoded), record, vocab_size=self.vocab_size,
                                        max_context=self.max_context)

    def decide(self, record: schema.Record, prepared: schema.Encoding) -> tuple[list[dict], dict]:
        self._owned()
        if self.used:
            raise ClefError("protocol_fault", "The Clef runtime permits one decision attempt.")
        self.used = True
        started = time.monotonic()
        replay = self.encode(record)
        if replay != prepared:
            raise ClefError("protocol_fault", "The decision encoding changed before inference.")
        if self.model.processor is not None:
            raise ClefError("protocol_fault", "The text-only decision processor changed.")
        with redirect_stdout(sys.stderr):
            actual, logits = self.model.logits(record.document(), max_length=self.max_context,
                                               max_state_tokens=None, truncate=False)
        checked = schema.validate_encoding(encoding_document(actual), record, vocab_size=self.vocab_size,
                                           max_context=self.max_context)
        if checked != prepared:
            raise ClefError("protocol_fault", "The decision encoding changed during inference.")
        if type(logits) not in (list, tuple) or len(logits) != len(prepared.questions):
            raise ClefError("protocol_fault", "The decision head returned an incomplete logit set.")
        probabilities = []
        for question, logit in zip(prepared.questions, logits):
            if tuple(logit.shape) != (len(question.option_ids),):
                raise ClefError("protocol_fault", "The decision logit shape differs from its options.")
            vector = self.mx.softmax(logit.astype(self.mx.float32))
            self.mx.eval(vector)
            probabilities.append({"question_id": question.question_id, "option_ids": list(question.option_ids),
                                  "values": vector.tolist()})
        schema.validate_probabilities(probabilities, prepared)
        self.mx.synchronize()
        metrics = {"elapsed_ms": int((time.monotonic() - started) * 1000),
                   "peak_device_bytes": self.mx.get_peak_memory()}
        return probabilities, metrics

    def close(self) -> None:
        self._owned()
        self.mx.synchronize()
        self.model = None
        gc.collect()
        self.mx.clear_cache()
        self.mx.synchronize()
