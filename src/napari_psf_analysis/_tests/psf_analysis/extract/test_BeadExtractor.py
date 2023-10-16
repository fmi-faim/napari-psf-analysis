import numpy as np
import pytest

from napari_psf_analysis.psf_analysis.extract.BeadExtractor import BeadExtractor
from napari_psf_analysis.psf_analysis.image import Calibrated3DImage

DATA = np.zeros((300, 200, 200), dtype=np.uint16)
DATA[50, 90, 110] = 1
SPACING = (200.0, 103.17, 103.17)
PATCH_SIZE = (6000, 2000, 2000)


@pytest.fixture
def bead_extractor():
    image = Calibrated3DImage(
        data=DATA,
        spacing=SPACING,
    )
    return BeadExtractor(image=image, patch_size=PATCH_SIZE)


def test_margins(bead_extractor):
    assert bead_extractor._margins[0] == PATCH_SIZE[0] / SPACING[0]
    assert bead_extractor._margins[1] == PATCH_SIZE[1] / SPACING[1]
    assert bead_extractor._margins[2] == PATCH_SIZE[2] / SPACING[2]


def test_in_margins(bead_extractor):
    point = (88.1, 43.7, 123.73)
    assert bead_extractor._in_margins(point=point)
    point = (12.3, 43.7, 123.73)
    assert not bead_extractor._in_margins(point=point)
    point = (88.3, 5.7, 123.73)
    assert not bead_extractor._in_margins(point=point)


def test_not_enough_margin(bead_extractor):
    points = bead_extractor.extract_beads(points=np.array([[1, 1, 1]]))

    assert len(points) == 0


def test_create_slices(bead_extractor):
    slices = bead_extractor._create_slices(point=(88.3, 43.7, 123.5))
    assert len(slices) == 3
    assert slices[0] == slice(73, 103)
    assert slices[1] == slice(34, 53)
    assert slices[2] == slice(114, 133)


def test_extract_rough_crop(bead_extractor):
    crop = bead_extractor._extract_rough_crop(point=(52.1, 87.5, 108.8))

    assert crop.shape == (30, 19, 19)
    assert crop[13, 12, 11] == 1


def test_compute_peak_offset(bead_extractor):
    crop = bead_extractor._extract_rough_crop(point=(52.1, 87.0, 108.8))
    offset = bead_extractor._compute_peak_offset(crop=crop)
    assert len(offset) == 3
    assert offset[0] == -2
    assert offset[1] == 3
    assert offset[2] == 2


def test_find_closest_peak(bead_extractor):
    closest_peak = bead_extractor._find_closest_peak(point=(52.1, 87.5, 108.8))
    assert closest_peak == (50, 90, 110)


def test_extract_beads(bead_extractor):
    beads = bead_extractor.extract_beads(points=np.array([[52.1, 87.5, 108.8]]))

    assert len(beads) == 1
    assert beads[0].data.shape == (30, 19, 19)
    assert beads[0].data[15, 9, 9] == 1
    assert beads[0].spacing == SPACING
    assert beads[0].offset == (50 - 15, 90 - 9, 110 - 9)

    start_z, start_y, start_x = beads[0].offset
    z_slice = slice(start_z, start_z + beads[0].data.shape[0])
    y_slice = slice(start_y, start_y + beads[0].data.shape[1])
    x_slice = slice(start_x, start_x + beads[0].data.shape[2])

    assert np.all(beads[0].data == DATA[z_slice, y_slice, x_slice])
