"""Prompt-construction (offline) + SentencePiece encode (gated) tests."""

from __future__ import annotations

import numpy as np
import pytest

from apxinf.processors import PromptTokenizer, build_prompt


def test_build_prompt_cleans_task_no_state():
    assert build_prompt("  pick_up the\nblock  ") == "pick up the block"


def test_build_prompt_state_template_matches_rust():
    # Rust pi05_prompt: "Task: {task}, State: {s0 s1 ...};\nAction: "
    state = [-1.0, 0.0, 1.0]  # -> 0, 128, 255
    prompt = build_prompt("open_drawer", state=state, discrete_state=True)
    assert prompt == "Task: open drawer, State: 0 128 255;\nAction: "


def test_build_prompt_state_required_when_discrete():
    with pytest.raises(ValueError):
        build_prompt("task", state=None, discrete_state=True)


def test_tokenizer_encodes_within_bounds(tokenizer_path):
    tok = PromptTokenizer(tokenizer_path)
    tokens = tok("pick up the red block")
    assert tokens.dtype == np.uint32
    assert 0 < tokens.size <= tok.max_token_len


def test_tokenizer_matches_reference_encode(tokenizer_path):
    import sentencepiece

    sp = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path)
    prompt = "pick_up the block\n"
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    want = list(sp.encode(cleaned, add_bos=True)) + list(sp.encode("\n"))

    got = PromptTokenizer(tokenizer_path)(prompt)
    np.testing.assert_array_equal(got, np.asarray(want, dtype=np.uint32))


def test_tokenizer_override_max_len(tokenizer_path):
    tok = PromptTokenizer(tokenizer_path)
    tight = tok.with_overrides(max_token_len=1)
    assert tight.max_token_len == 1 and tok.max_token_len == 200  # original untouched
    with pytest.raises(ValueError):
        tight("this prompt is definitely longer than one token")
