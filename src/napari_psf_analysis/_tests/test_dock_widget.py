from os.path import exists
from pathlib import Path

import pytest
import yaml

from napari_psf_analysis._dock_widget import (
    PsfAnalysis,
    get_dpi,
    get_microscopes,
    get_output_path,
)

OUTPUT_PATH = "/a/path/to/a/dir"
MICROSCOPES = ["mic1", "mic2"]
DPI = ["96", "150", "300"]


@pytest.fixture
def path(tmp_path_factory):
    return tmp_path_factory.mktemp("napari-psf-analysis") / "config.yaml"


@pytest.fixture
def add_output_path(path):
    if exists(path):
        with open(path, "a") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    config["output_path"] = OUTPUT_PATH
    with open(path, "w") as f:
        yaml.dump(config, f)


@pytest.fixture
def add_microscopes(path):
    if exists(path):
        with open(path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    config["microscopes"] = MICROSCOPES
    with open(path, "w") as f:
        yaml.dump(config, f)


@pytest.fixture
def add_dpi(path):
    if exists(path):
        with open(path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    config["dpi"] = DPI
    with open(path, "w") as f:
        yaml.dump(config, f)


def test_load_config(path, add_output_path, add_microscopes, add_dpi):
    microscopes = get_microscopes(path)
    assert microscopes == MICROSCOPES

    dpi = get_dpi(path)
    assert dpi == DPI

    output_path = get_output_path(path)
    assert output_path == Path(OUTPUT_PATH)


def test_get_default_config(path):
    microscopes = get_microscopes(path)
    assert microscopes == "Microscope"

    dpi = get_dpi(path)
    assert dpi == "150"

    output_path = get_output_path(path)
    assert output_path == Path.home()


def test_load_from_partial_config_1(path, add_microscopes):
    microscopes = get_microscopes(path)
    assert microscopes == MICROSCOPES

    dpi = get_dpi(path)
    assert dpi == "150"

    output_path = get_output_path(path)
    assert output_path == Path.home()


def test_load_from_partial_config_2(path, add_output_path, add_dpi):
    microscopes = get_microscopes(path)
    assert microscopes == "Microscope"

    dpi = get_dpi(path)
    assert dpi == DPI

    output_path = get_output_path(path)
    assert output_path == Path(OUTPUT_PATH)


def test_widget_creation(make_napari_viewer_proxy):
    widget = PsfAnalysis(napari_viewer=make_napari_viewer_proxy())

    assert widget is not None
