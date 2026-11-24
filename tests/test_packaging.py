from pathlib import Path

import tomllib


ROOT = Path(__file__).parents[1]


def test_optional_provider_extras_are_declared_in_project_metadata():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    optional = project["optional-dependencies"]

    assert optional["api"] == ["langchain-anthropic>=0.3"]
    assert optional["vertex"] == ["langchain-google-vertexai>=2.0"]
    assert optional["langfuse"] == ["langfuse>=2.0"]


def test_wheel_metadata_contains_provider_extra_requirements(tmp_path, monkeypatch):
    """Exercise setuptools' metadata generation, rather than a checked-in artifact."""
    import setuptools.build_meta as build_meta

    monkeypatch.chdir(ROOT)
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    dist_info = Path(build_meta.prepare_metadata_for_build_wheel(str(metadata_dir)))
    metadata = (metadata_dir / dist_info / "METADATA").read_text()

    assert 'Requires-Dist: langchain-anthropic>=0.3; extra == "api"' in metadata
    assert 'Requires-Dist: langchain-google-vertexai>=2.0; extra == "vertex"' in metadata
    assert 'Requires-Dist: langfuse>=2.0; extra == "langfuse"' in metadata
