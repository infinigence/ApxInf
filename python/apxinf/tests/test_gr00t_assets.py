"""Offline tests for GR00T's relocatable configuration/processor asset bundle."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

_HELPER = Path(__file__).resolve().parents[1] / "apxinf/policies/impls/_gr00t_assets.py"
_SPEC = importlib.util.spec_from_file_location("_gr00t_assets_test_subject", _HELPER)
assert _SPEC is not None and _SPEC.loader is not None
assets = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(assets)


class Gr00tAssetsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "gr00t"
        self.source = self.root / "cosmos"
        self.model.mkdir()
        self.source.mkdir()
        for name in (
            "config.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "tokenizer.json",
        ):
            (self.source / name).write_bytes(b"{}\n")

    def prepare(self):
        return assets.prepare_assets(self.model, self.source)

    def edit_manifest(self, root, edit):
        path = root / assets.MANIFEST
        document = json.loads(path.read_text())
        edit(document)
        path.write_text(json.dumps(document), encoding="utf-8")

    def test_prepare_copies_resources_and_templates_but_no_weights_or_cache(self):
        (self.source / "video_preprocessor_config.json").write_bytes(b'{"size": 256}\n')
        (self.source / "chat_templates").mkdir()
        (self.source / "chat_templates" / "default.jinja").write_text("{{ messages }}\n")
        (self.source / "model.safetensors").write_bytes(b"weights must stay at source")
        (self.source / "model.safetensors.index.json").write_text("{}")
        (self.source / "README.md").write_text("snapshot metadata")
        (self.source / ".cache").mkdir()
        (self.source / ".cache" / "tokenizer.json").write_text("cached unrelated file")

        destination = self.prepare()

        self.assertEqual(destination, self.model / "assets" / "cosmos")
        self.assertEqual(assets.resolve_assets(self.model), destination)
        files = assets.verify_asset_manifest(destination)
        self.assertEqual(
            set(files),
            {
                "config.json", "tokenizer_config.json", "preprocessor_config.json",
                "tokenizer.json", "video_preprocessor_config.json", "chat_templates/default.jinja",
            },
        )
        for relative in files:
            self.assertEqual((destination / relative).read_bytes(), (self.source / relative).read_bytes())
        for excluded in ("model.safetensors", "model.safetensors.index.json", "README.md", ".cache"):
            self.assertFalse((destination / excluded).exists())

    def test_prepare_materializes_huggingface_snapshot_file_symlinks(self):
        blob = self.root / "shared-tokenizer-blob"
        blob.write_bytes(b'{"vocab": {}}\n')
        (self.source / "tokenizer.json").unlink()
        (self.source / "tokenizer.json").symlink_to(blob)

        destination = self.prepare()

        self.assertFalse((destination / "tokenizer.json").is_symlink())
        self.assertEqual((destination / "tokenizer.json").read_bytes(), blob.read_bytes())
        blob.unlink()
        assets.verify_asset_manifest(destination)

    def test_vocab_and_merges_can_replace_tokenizer_json(self):
        (self.source / "tokenizer.json").unlink()
        (self.source / "vocab.json").write_text("{}")
        (self.source / "merges.txt").write_text("#version: 0.2\n")
        files = assets.verify_asset_manifest(self.prepare())
        self.assertNotIn("tokenizer.json", files)
        self.assertTrue({"vocab.json", "merges.txt"} <= files.keys())

    def test_prepare_rejects_missing_required_file_before_creating_destination(self):
        (self.source / "preprocessor_config.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing required.*preprocessor_config"):
            self.prepare()
        self.assertFalse((self.model / assets.DEFAULT_SUBDIR).exists())

    def test_prepare_rejects_partial_tokenizer_fallback(self):
        (self.source / "tokenizer.json").unlink()
        (self.source / "vocab.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "both vocab.json and merges.txt"):
            self.prepare()
        self.assertFalse((self.model / assets.DEFAULT_SUBDIR).exists())

    def test_existing_destination_is_not_overwritten(self):
        destination = self.model / assets.DEFAULT_SUBDIR
        destination.mkdir(parents=True)
        marker = destination / "user-file"
        marker.write_bytes(b"keep this directory")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(marker.read_bytes(), b"keep this directory")
        self.assertEqual(list(destination.iterdir()), [marker])

    def test_default_directory_requires_manifest(self):
        shutil.copytree(self.source, self.model / assets.DEFAULT_SUBDIR)
        with self.assertRaisesRegex(FileNotFoundError, "Gr00tPolicy.prepare_assets"):
            assets.resolve_assets(self.model)

    def test_explicit_legacy_snapshot_overrides_default_without_new_validation(self):
        destination = self.prepare()
        (destination / "config.json").write_text("corrupted default")
        self.assertEqual(assets.resolve_assets(self.model, self.source), self.source)
        missing_legacy = self.root / "caller-validated-legacy-path"
        self.assertEqual(assets.resolve_assets(self.model, missing_legacy), missing_legacy)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            assets.resolve_assets(self.model)

    def test_explicit_v1_bundle_cannot_bypass_manifest_validation(self):
        destination = self.prepare()
        (destination / "tokenizer.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch.*tokenizer.json"):
            assets.resolve_assets(self.model, destination)

    def test_manifest_rejects_unknown_schema_and_non_object_files(self):
        destination = self.prepare()
        original = (destination / assets.MANIFEST).read_bytes()
        for edit, message in (
            (lambda doc: doc.update(schema="apxinf.gr00t-assets.v2"), "schema"),
            (lambda doc: doc.update(files=[]), "files must be an object"),
            (lambda doc: doc.update(source="unversioned-provenance"), "only schema and files"),
        ):
            with self.subTest(message=message):
                (destination / assets.MANIFEST).write_bytes(original)
                self.edit_manifest(destination, edit)
                with self.assertRaisesRegex(ValueError, message):
                    assets.verify_asset_manifest(destination)

    def test_manifest_rejects_invalid_or_non_resource_paths(self):
        destination = self.prepare()
        original = (destination / assets.MANIFEST).read_bytes()
        for relative in (
            "../config.json", "/config.json", "./config.json", "chat_templates/../evil.jinja",
            "chat_templates//evil.jinja", "chat_templates\\evil.jinja", "config.json\0",
            "model.safetensors", "chat_templates/nested/evil.jinja", "",
        ):
            with self.subTest(relative=relative):
                (destination / assets.MANIFEST).write_bytes(original)
                self.edit_manifest(destination, lambda doc: doc["files"].update({relative: "0" * 64}))
                with self.assertRaisesRegex(ValueError, "asset path"):
                    assets.verify_asset_manifest(destination)

    def test_manifest_rejects_noncanonical_hashes(self):
        destination = self.prepare()
        original = (destination / assets.MANIFEST).read_bytes()
        for digest in ("A" * 64, "0" * 63, "sha256:" + "0" * 64, 123, "0" * 64 + "\n"):
            with self.subTest(digest=digest):
                (destination / assets.MANIFEST).write_bytes(original)
                self.edit_manifest(destination, lambda doc: doc["files"].update({"config.json": digest}))
                with self.assertRaisesRegex(ValueError, "invalid lowercase SHA256"):
                    assets.verify_asset_manifest(destination)

    def test_manifest_cannot_omit_required_file_even_if_file_is_absent(self):
        destination = self.prepare()
        (destination / "config.json").unlink()
        self.edit_manifest(destination, lambda doc: doc["files"].pop("config.json"))
        with self.assertRaisesRegex(ValueError, "missing required.*config.json"):
            assets.verify_asset_manifest(destination)

    def test_missing_and_modified_files_are_rejected(self):
        destination = self.prepare()
        path = destination / "config.json"
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "file is missing.*config.json"):
            assets.verify_asset_manifest(destination)
        path.write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch.*config.json"):
            assets.asset_identity(destination)

    def test_extra_resource_is_rejected_but_readme_and_cache_are_ignored(self):
        destination = self.prepare()
        (destination / "README.md").write_text("local notes")
        (destination / ".cache").mkdir()
        (destination / ".cache" / "config.json").write_text("unrelated")
        assets.verify_asset_manifest(destination)
        (destination / "video_preprocessor_config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "unlisted.*video_preprocessor_config"):
            assets.verify_asset_manifest(destination)

    def test_extra_chat_template_is_rejected(self):
        destination = self.prepare()
        (destination / "chat_templates").mkdir()
        (destination / "chat_templates" / "unexpected.jinja").write_text("hello")
        with self.assertRaisesRegex(ValueError, "unlisted.*unexpected.jinja"):
            assets.verify_asset_manifest(destination)

    def test_manifest_and_resource_symlinks_are_rejected(self):
        destination = self.prepare()
        for name in (assets.MANIFEST, "config.json"):
            with self.subTest(name=name):
                path = destination / name
                content = path.read_bytes()
                outside = self.root / "outside"
                outside.write_bytes(content)
                path.unlink()
                path.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                    assets.verify_asset_manifest(destination)
                path.unlink()
                path.write_bytes(content)

    def test_chat_template_directory_symlink_is_rejected(self):
        destination = self.prepare()
        external = self.root / "external-templates"
        external.mkdir()
        (destination / "chat_templates").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "directory must not be a symlink"):
            assets.verify_asset_manifest(destination)

    def test_duplicate_manifest_keys_are_rejected(self):
        destination = self.prepare()
        path = destination / assets.MANIFEST
        text = path.read_text()
        text = text.replace('"schema":', '"schema": "ignored", "schema":', 1)
        path.write_text(text)
        with self.assertRaisesRegex(ValueError, "duplicate.*schema"):
            assets.verify_asset_manifest(destination)

    def test_identity_is_independent_of_location_and_manifest_format(self):
        (self.source / "chat_templates").mkdir()
        (self.source / "chat_templates" / "é.jinja").write_text("{{ messages }}", encoding="utf-8")
        destination = self.prepare()
        copied = self.root / "relocated-cosmos"
        shutil.copytree(destination, copied)
        path = copied / assets.MANIFEST
        document = json.loads(path.read_text())
        document["files"] = dict(reversed(list(document["files"].items())))
        path.write_text(json.dumps(document, ensure_ascii=True), encoding="utf-8")
        self.assertEqual(assets.asset_identity(destination), assets.asset_identity(copied))
        self.assertRegex(assets.asset_identity(copied), r"^sha256:[0-9a-f]{64}$")

    def test_identity_matches_independent_v1_byte_protocol(self):
        # This fixture has four identical files. Spell out the wire bytes in
        # their specified order instead of using the helper's manifest/sort code.
        file_hash = hashlib.sha256(b"{}\n").hexdigest().encode("ascii")
        preimage = (
            b"apxinf.gr00t-assets.v1\0"
            + b"config.json\0" + file_hash + b"\0"
            + b"preprocessor_config.json\0" + file_hash + b"\0"
            + b"tokenizer.json\0" + file_hash + b"\0"
            + b"tokenizer_config.json\0" + file_hash + b"\0"
        )
        expected = "sha256:" + hashlib.sha256(preimage).hexdigest()
        self.assertEqual(assets.asset_identity(self.prepare()), expected)

if __name__ == "__main__":
    unittest.main()
