import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "upload_modelscope_archive.py"
SPEC = importlib.util.spec_from_file_location("upload_modelscope_archive", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
archive_label = MODULE.archive_label
clean_prefix = MODULE.clean_prefix
next_index = MODULE.next_index
preview_files = MODULE.preview_files
list_files = MODULE.list_files


def test_archive_label_removes_generator_run_prefix():
    path = Path("20260909_010203_00000_StackYellowOnRedTask_seed123.tar.gz")
    assert archive_label(path) == "StackYellowOnRedTask_seed123"


def test_next_index_ignores_other_remote_layouts():
    paths = [
        "README.md",
        "data/000000_FirstTask_seed1.tar.gz",
        "data/000003_SecondTask_seed2.tar.gz",
        "data/legacy/example.tar.gz",
        "samples/000100_not_data.tar.gz",
    ]
    assert next_index(paths, "data", 6) == 4


def test_clean_prefix_rejects_parent_components():
    try:
        clean_prefix("data/../bad")
    except ValueError:
        pass
    else:
        raise AssertionError("parent path component was accepted")


def test_preview_files_selects_supported_images(tmp_path):
    for name in ("b.jpg", "a.png", "ignored.txt"):
        (tmp_path / name).write_bytes(b"preview")
    assert [path.name for path in preview_files(tmp_path)] == ["a.png", "b.jpg"]


def test_list_files_uses_paginated_dataset_endpoint():
    class Legacy:
        def list_dataset_files_paginated(self, **kwargs):
            assert kwargs["page_size"] == 200
            return [
                {"Path": "data/000123_demo.tar.gz", "Size": 42},
                {"Path": "preview/000123_demo/frame.png", "Size": 7},
            ]

    class Api:
        legacy = Legacy()

        def list_repo_files(self, **kwargs):
            raise AssertionError("unpaginated endpoint must not be used")

    files = list_files(Api(), "owner/dataset", "master")
    assert [(entry.path, entry.size) for entry in files] == [
        ("data/000123_demo.tar.gz", 42),
        ("preview/000123_demo/frame.png", 7),
    ]
