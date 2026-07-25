"""Every registered method has a complete, non-drifting doc page (DOC-04).

This is the check that keeps `docs/methods/<name>.md` honest as methods and their configs
evolve. It asserts, for **every method in the registry** (so a method added later is covered
with no edit here):

* the doc page exists;
* it carries the required sections — algorithm, explicit pre/post-processing, a config
  reference, known failure modes, and a mirrored ROBUSTNESS BACKLOG;
* **every field of the method's `config_model` JSON Schema appears in the doc.** The config
  model is the single source of truth (it also drives the UI form), so a field that is added
  to the model but not documented is drift, and this test fails until the doc is updated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from object_search.provenance import repo_root
from object_search.search.registry import MethodSpec, list_methods

# Substrings (case-insensitive) that every method page must contain. Substrings rather than
# exact headers so a method may specialise a heading (e.g. sparse-geo splits pre-processing per
# backend) without escaping the check.
_REQUIRED_SECTIONS = (
    "algorithm",
    "pre-processing",
    "post-processing",
    "config reference",
    "failure modes",
    "robustness backlog",
)


def _methods() -> list[MethodSpec]:
    return sorted(list_methods(), key=lambda spec: spec.name)


def _doc_path(name: str) -> Path:
    return repo_root() / "docs" / "methods" / f"{name}.md"


@pytest.mark.parametrize("spec", _methods(), ids=lambda s: s.name)
def test_method_doc_exists(spec: MethodSpec) -> None:
    """Every registered method has a doc page on disk."""
    assert _doc_path(spec.name).is_file(), f"missing docs/methods/{spec.name}.md"


@pytest.mark.parametrize("spec", _methods(), ids=lambda s: s.name)
def test_method_doc_has_required_sections(spec: MethodSpec) -> None:
    """Each doc carries every required section."""
    text = _doc_path(spec.name).read_text(encoding="utf-8").lower()
    missing = [section for section in _REQUIRED_SECTIONS if section not in text]
    assert not missing, f"{spec.name}.md missing sections: {missing}"


@pytest.mark.parametrize("spec", _methods(), ids=lambda s: s.name)
def test_config_reference_covers_every_schema_field(spec: MethodSpec) -> None:
    """Every config-model JSON Schema field is named in the doc (no config drift, DOC-04)."""
    text = _doc_path(spec.name).read_text(encoding="utf-8")
    properties = spec.config_model.model_json_schema().get("properties", {})
    assert properties, f"{spec.name} has no config properties to document"
    missing = [field for field in properties if field not in text]
    assert not missing, (
        f"{spec.name}.md does not document config field(s) {missing}; "
        f"regenerate the config reference from config_model.model_json_schema()"
    )
