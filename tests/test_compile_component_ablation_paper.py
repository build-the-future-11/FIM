from pathlib import Path

import pytest

from scripts.compile_component_ablation_paper import validated_source


def test_compile_gate_rejects_incomplete_markers(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    path.write_text("# Paper\n\nResults pending. " + "word " * 2_100)
    with pytest.raises(ValueError, match="incomplete"):
        validated_source(path)


def test_compile_gate_accepts_substantive_titled_source(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    path.write_text("# Evidence Paper\n\n" + "evidence " * 2_100)
    title, body = validated_source(path)
    assert title == "Evidence Paper"
    assert len(body.split()) == 2_100
