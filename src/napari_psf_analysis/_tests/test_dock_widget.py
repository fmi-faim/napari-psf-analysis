import pathlib
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import yaml
from numpy.testing import assert_almost_equal
from qtpy.QtWidgets import QLineEdit
from tifffile import imsave

from napari_psf_analysis._dock_widget import (
    PsfAnalysis,
    get_microscopes,
    get_output_path,
)
from napari_psf_analysis.utils.gaussians import gaussian_3d


def test_psf_analysis_widget_discard_point(make_napari_viewer, tmpdir, capsys):
    viewer = make_napari_viewer()
    height = 875
    z, y, x = 20, 20, 20
    sigma_z, sigma_y, sigma_x = 2, 1, 2
    offset = 150

    Z, Y, X = np.meshgrid(np.arange(40), np.arange(40), np.arange(40), indexing="ij")
    gaussian = gaussian_3d(height, z, y, x, sigma_z, sigma_y, sigma_x, offset)

    img = gaussian(Z, Y, X)

    tmp_path = tmpdir.mkdir("test").join("data.tif")
    imsave(tmp_path, img)
    viewer.open(str(tmp_path))
    viewer.add_points(np.array([[z, y, x]]))

    widget = PsfAnalysis(viewer)

    widget.extract_psfs.click()

    captured = capsys.readouterr()
    assert (
        captured.out
        == "INFO: Discarded point (20, 20, 20). Too close to image border.\n"
    )


def test_psf_analysis_widget_basic(make_napari_viewer, tmpdir, capsys):
    viewer = make_napari_viewer()
    height = 875
    z, y, x = 20, 20, 20
    sigma_z, sigma_y, sigma_x = 2, 1, 2
    offset = 150

    Z, Y, X = np.meshgrid(np.arange(40), np.arange(40), np.arange(40), indexing="ij")
    gaussian = gaussian_3d(height, z, y, x, sigma_z, sigma_y, sigma_x, offset)

    img = gaussian(Z, Y, X)

    tmp_path = tmpdir.mkdir("test")
    input_path = tmp_path.join("data.tif")
    imsave(input_path, img)
    viewer.open(str(input_path))
    viewer.add_points(np.array([[z, y, x]]))

    widget = PsfAnalysis(viewer)
    widget.xy_pixelsize.setValue(1)
    widget.z_spacing.setValue(1)
    widget.psf_box_size.setValue(30)

    widget.microscope = QLineEdit("Microscope")
    widget.extract_psfs.click()

    captured = capsys.readouterr()
    assert captured.out == ""

    assert len(viewer.layers) == 3
    assert viewer.layers[-1].data.shape == (1, 1000, 1000, 3)

    widget.save_dir_line_edit.setText(str(tmp_path))
    widget.save_button.click()

    psf_img_path = glob(str(tmp_path.join("*.png")))[0]
    assert psf_img_path == str(tmp_path.join("data.tif_Bead_X20.0_Y20.0_Z20.0.png"))

    csv_path = glob(str(tmp_path.join("*.csv")))[0]
    assert csv_path == str(
        tmp_path.join(
            "PSFMeasurement_{}_data.tif_Microscope_100.0_1.4.csv".format(
                datetime.today().strftime("%Y-%m-%d")
            )
        )
    )

    results = pd.read_csv(str(csv_path))
    assert len(results.columns) == 18
    assert results.columns[0] == "ImageName"
    assert results.columns[1] == "Date"
    assert results.columns[2] == "Microscope"
    assert results.columns[3] == "Magnification"
    assert results.columns[4] == "NA"
    assert results.columns[5] == "X"
    assert results.columns[6] == "Y"
    assert results.columns[7] == "Z"
    assert results.columns[8] == "FWHM_X"
    assert results.columns[9] == "FWHM_Y"
    assert results.columns[10] == "FWHM_Z"
    assert results.columns[11] == "r2_x"
    assert results.columns[12] == "r2_y"
    assert results.columns[13] == "r2_z"
    assert results.columns[14] == "SignalToBG"
    assert results.columns[15] == "XYpixelsize"
    assert results.columns[16] == "Zspacing"
    assert results.columns[17] == "PSF_path"

    assert results.iloc[0][0] == "data.tif"
    assert results.iloc[0][1] == datetime.today().strftime("%Y-%m-%d")
    assert results.iloc[0][2] == "Microscope"
    assert results.iloc[0][3] == 100.0
    assert results.iloc[0][4] == 1.4
    assert results.iloc[0][5] == 20.0
    assert results.iloc[0][6] == 20.0
    assert results.iloc[0][7] == 20.0
    assert_almost_equal(results.iloc[0][8], sigma_x * 2 * np.sqrt(2 * np.log(2)))
    assert_almost_equal(results.iloc[0][9], sigma_y * 2 * np.sqrt(2 * np.log(2)))
    assert_almost_equal(results.iloc[0][10], sigma_z * 2 * np.sqrt(2 * np.log(2)))
    assert results.iloc[0][11] == 1.0
    assert results.iloc[0][12] == 1.0
    assert results.iloc[0][13] == 1.0
    assert_almost_equal(results.iloc[0][14], np.mean(img[20, 19:22, 19:22]) / offset)
    assert results.iloc[0][15] == 1.0
    assert results.iloc[0][16] == 1.0
    assert results.iloc[0][17] == "./data.tif_Bead_X20.0_Y20.0_Z20.0.png"

    # Remove measured psf
    widget.delete_measurement.click()
    assert len(viewer.layers) == 2


def test_psf_analysis_widget_advanced(make_napari_viewer, tmpdir, capsys):
    viewer = make_napari_viewer()
    height = 875
    z, y, x = 20, 20, 20
    sigma_z, sigma_y, sigma_x = 2, 1, 2
    offset = 150

    Z, Y, X = np.meshgrid(np.arange(40), np.arange(40), np.arange(40), indexing="ij")
    gaussian = gaussian_3d(height, z, y, x, sigma_z, sigma_y, sigma_x, offset)

    img = gaussian(Z, Y, X)

    tmp_path = tmpdir.mkdir("test")
    input_path = tmp_path.join("data.tif")
    imsave(input_path, img)
    viewer.open(str(input_path))
    viewer.add_points(np.array([[z, y, x]]))

    widget = PsfAnalysis(viewer)
    widget.xy_pixelsize.setValue(1)
    widget.z_spacing.setValue(1)
    widget.psf_box_size.setValue(30)

    widget.temperature.setValue(25)
    widget.airy_unit.setValue(1)
    widget.bead_size.setValue(3)
    widget.bead_supplier.setText("Beads by Dr. Gelman")
    widget.mounting_medium.setText("Mountain Dew")
    widget.objective_id.setText("no-ID")
    widget.operator.setText("Tester")
    widget.microscope_type.setText("Simulation")
    widget.excitation.setValue(405)
    widget.emission.setValue(568)
    widget.comment.setText("I have a question.")

    widget.microscope = QLineEdit("Microscope")
    widget.extract_psfs.click()

    captured = capsys.readouterr()
    assert captured.out == ""

    assert len(viewer.layers) == 3
    assert viewer.layers[-1].data.shape == (1, 1000, 1000, 3)

    widget.save_dir_line_edit.setText(str(tmp_path))
    widget.save_button.click()

    psf_img_path = glob(str(tmp_path.join("*.png")))[0]
    assert psf_img_path == str(tmp_path.join("data.tif_Bead_X20.0_Y20.0_Z20.0.png"))

    csv_path = glob(str(tmp_path.join("*.csv")))[0]
    assert csv_path == str(
        tmp_path.join(
            "PSFMeasurement_{}_data.tif_Microscope_100.0_1.4.csv".format(
                datetime.today().strftime("%Y-%m-%d")
            )
        )
    )

    results = pd.read_csv(str(csv_path))
    assert len(results.columns) == 29
    assert results.columns[0] == "ImageName"
    assert results.columns[1] == "Date"
    assert results.columns[2] == "Microscope"
    assert results.columns[3] == "Magnification"
    assert results.columns[4] == "NA"
    assert results.columns[5] == "X"
    assert results.columns[6] == "Y"
    assert results.columns[7] == "Z"
    assert results.columns[8] == "FWHM_X"
    assert results.columns[9] == "FWHM_Y"
    assert results.columns[10] == "FWHM_Z"
    assert results.columns[11] == "r2_x"
    assert results.columns[12] == "r2_y"
    assert results.columns[13] == "r2_z"
    assert results.columns[14] == "SignalToBG"
    assert results.columns[15] == "XYpixelsize"
    assert results.columns[16] == "Zspacing"
    assert results.columns[17] == "Temperature"
    assert results.columns[18] == "AiryUnit"
    assert results.columns[19] == "BeadSize"
    assert results.columns[20] == "BeadSupplier"
    assert results.columns[21] == "MountingMedium"
    assert results.columns[22] == "Objective_id"
    assert results.columns[23] == "Operator"
    assert results.columns[24] == "MicroscopeType"
    assert results.columns[25] == "Excitation"
    assert results.columns[26] == "Emission"
    assert results.columns[27] == "Comment"
    assert results.columns[28] == "PSF_path"

    assert results.iloc[0][0] == "data.tif"
    assert results.iloc[0][1] == datetime.today().strftime("%Y-%m-%d")
    assert results.iloc[0][2] == "Microscope"
    assert results.iloc[0][3] == 100.0
    assert results.iloc[0][4] == 1.4
    assert results.iloc[0][5] == 20.0
    assert results.iloc[0][6] == 20.0
    assert results.iloc[0][7] == 20.0
    assert_almost_equal(results.iloc[0][8], sigma_x * 2 * np.sqrt(2 * np.log(2)))
    assert_almost_equal(results.iloc[0][9], sigma_y * 2 * np.sqrt(2 * np.log(2)))
    assert_almost_equal(results.iloc[0][10], sigma_z * 2 * np.sqrt(2 * np.log(2)))
    assert results.iloc[0][11] == 1.0
    assert results.iloc[0][12] == 1.0
    assert results.iloc[0][13] == 1.0
    assert_almost_equal(results.iloc[0][14], np.mean(img[20, 19:22, 19:22]) / offset)
    assert results.iloc[0][15] == 1.0
    assert results.iloc[0][16] == 1.0
    assert results.iloc[0][17] == 25
    assert results.iloc[0][18] == 1
    assert results.iloc[0][19] == 3
    assert results.iloc[0][20] == "Beads by Dr. Gelman"
    assert results.iloc[0][21] == "Mountain Dew"
    assert results.iloc[0][22] == "no-ID"
    assert results.iloc[0][23] == "Tester"
    assert results.iloc[0][24] == "Simulation"
    assert results.iloc[0][25] == 405
    assert results.iloc[0][26] == 568
    assert results.iloc[0][27] == "I have a question."
    assert results.iloc[0][28] == "./data.tif_Bead_X20.0_Y20.0_Z20.0.png"


def test_set_config(tmpdir):
    config_dir = tmpdir.mkdir("config_dir")

    save_dir = tmpdir.mkdir("save_dir")
    settings = {"microscopes": ["Test1", "Test2"], "output_path": str(save_dir)}

    config_name = config_dir.join("psf_analyze_settings.yaml")
    with open(config_name, "w") as f:
        yaml.dump(settings, f)

    microscopes = get_microscopes(psf_settings_path=None)
    assert microscopes == "Microscope"
    microscopes = get_microscopes(psf_settings_path=str(config_name))
    assert microscopes == ["Test1", "Test2"]

    outpath = get_output_path(psf_settings_path=None)
    assert outpath == pathlib.Path.home()
    outpath = get_output_path(psf_settings_path=str(config_name))
    assert str(outpath) == str(save_dir)
