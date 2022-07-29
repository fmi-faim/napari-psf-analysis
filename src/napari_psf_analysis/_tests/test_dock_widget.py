# import pathlib
# from datetime import datetime
# from glob import glob
#
# import numpy as np
# import pandas as pd
# import yaml
# from numpy.testing import assert_almost_equal
# from qtpy.QtWidgets import QLineEdit
# from tifffile import imwrite
#
# from napari_psf_analysis._dock_widget import (
#     PsfAnalysis,
#     get_microscopes,
#     get_output_path,
# )
# from napari_psf_analysis.utils.gaussians import gaussian_3d
#
#
# def test_psf_analysis_widget_discard_point(make_napari_viewer, tmpdir, capsys):
#     viewer = make_napari_viewer()
#     height = 875
#     z, y, x = 20, 20, 20
#     sigma_z, sigma_y, sigma_x = 2, 1, 2
#     offset = 150
#
#     Z, Y, X = np.meshgrid(np.arange(40), np.arange(40), np.arange(40), indexing="ij")
#     gaussian = gaussian_3d(height, z, y, x, sigma_z, sigma_y, sigma_x, offset)
#
#     img = gaussian(Z, Y, X)
#
#     tmp_path = tmpdir.mkdir("test").join("data.tif")
#     imwrite(tmp_path, img)
#     viewer.open(str(tmp_path))
#     viewer.add_points(np.array([[z, y, x]]))
#
#     widget = PsfAnalysis(viewer)
#
#     widget.extract_psfs.click()
#
#     captured = capsys.readouterr()
#     assert (
#         captured.out
#         == "INFO: Discarded point (20, 20, 20). Too close to image border.\n"
#     )
#
#
# def test_psf_analysis_widget_basic(make_napari_viewer, tmpdir, capsys):
#     viewer = make_napari_viewer()
#     height = 875
#     z, y, x = 20, 20.2, 20.5
#     sigma_z, sigma_y, sigma_x = 5, 2, 1
#     offset = 150
#
#     Z, Y, X = np.meshgrid(np.arange(40), np.arange(40), np.arange(40), indexing="ij")
#     gaussian = gaussian_3d(height, z, y, x, sigma_z, sigma_y, sigma_x, offset)
#
#     img = gaussian(Z, Y, X)
#
#     tmp_path = tmpdir.mkdir("test")
#     input_path = tmp_path.join("data.tif")
#     imwrite(input_path, img)
#     viewer.open(str(input_path))
#     viewer.add_points(np.array([[z, y, x]]))
#
#     widget = PsfAnalysis(viewer)
#     widget.xy_pixelsize.setValue(1)
#     widget.z_spacing.setValue(1)
#     widget.psf_box_size.setValue(30)
#
#     widget.microscope = QLineEdit("Microscope")
#     widget.extract_psfs.click()
#
#     captured = capsys.readouterr()
#     assert captured.out == ""
#
#     assert len(viewer.layers) == 3
#     assert viewer.layers[-1].data.shape == (1, 1000, 1000, 3)
#
#     widget.save_dir_line_edit.setText(str(tmp_path))
#     widget.save_button.click()
#
#     psf_img_path = glob(str(tmp_path.join("*.png")))[0]
#     assert psf_img_path == str(tmp_path.join("data.tif_Bead_X20.5_Y20.2_Z20.0.png"))
#
#     csv_path = glob(str(tmp_path.join("*.csv")))[0]
#     assert csv_path == str(
#         tmp_path.join(
#             "PSFMeasurement_{}_data.tif_Microscope_100_1.4.csv".format(
#                 datetime.today().strftime("%Y-%m-%d")
#             )
#         )
#     )
#
#     results = pd.read_csv(str(csv_path))
#     assert len(results.columns) == 37
#     assert results.columns[0] == "ImageName"
#     assert results.columns[1] == "Date"
#     assert results.columns[2] == "Microscope"
#     assert results.columns[3] == "Magnification"
#     assert results.columns[4] == "NA"
#     assert results.columns[5] == "Amplitude"
#     assert results.columns[6] == "Background"
#     assert results.columns[7] == "X"
#     assert results.columns[8] == "Y"
#     assert results.columns[9] == "Z"
#     assert results.columns[10] == "FWHM_X"
#     assert results.columns[11] == "FWHM_Y"
#     assert results.columns[12] == "FWHM_Z"
#     assert results.columns[13] == "PrincipalAxis_1"
#     assert results.columns[14] == "PrincipalAxis_2"
#     assert results.columns[15] == "PrincipalAxis_3"
#     assert results.columns[16] == "SignalToBG"
#     assert results.columns[17] == "XYpixelsize"
#     assert results.columns[18] == "Zspacing"
#     assert results.columns[19] == "cov_xx"
#     assert results.columns[20] == "cov_xy"
#     assert results.columns[21] == "cov_xz"
#     assert results.columns[22] == "cov_yy"
#     assert results.columns[23] == "cov_yz"
#     assert results.columns[24] == "cov_zz"
#     assert results.columns[25] == "sde_peak"
#     assert results.columns[26] == "sde_background"
#     assert results.columns[27] == "sde_X"
#     assert results.columns[28] == "sde_Y"
#     assert results.columns[29] == "sde_Z"
#     assert results.columns[30] == "sde_cov_xx"
#     assert results.columns[31] == "sde_cov_xy"
#     assert results.columns[32] == "sde_cov_xz"
#     assert results.columns[33] == "sde_cov_yy"
#     assert results.columns[34] == "sde_cov_yz"
#     assert results.columns[35] == "sde_cov_zz"
#     assert results.columns[36] == "PSF_path"
#
#     assert results.iloc[0][0] == "data.tif"
#     assert results.iloc[0][1] == datetime.today().strftime("%Y-%m-%d")
#     assert results.iloc[0][2] == "Microscope"
#     assert results.iloc[0][3] == 100.0
#     assert results.iloc[0][4] == 1.4
#     assert_almost_equal(results.iloc[0][5], height, decimal=0)
#     assert_almost_equal(results.iloc[0][6], offset, decimal=0)
#     assert_almost_equal(results.iloc[0][7], 20.5)
#     assert_almost_equal(results.iloc[0][8], 20.2)
#     assert_almost_equal(results.iloc[0][9], 20.0)
#
#     def fwhm(x):
#         return 2 * np.sqrt(2 * np.log(2)) * x
#
#     assert_almost_equal(results.iloc[0][10], fwhm(sigma_x))
#     assert_almost_equal(results.iloc[0][11], fwhm(sigma_y))
#     assert_almost_equal(results.iloc[0][12], fwhm(sigma_z))
#     assert_almost_equal(results.iloc[0][13], fwhm(sigma_z))
#     assert_almost_equal(results.iloc[0][14], fwhm(sigma_y))
#     assert_almost_equal(results.iloc[0][15], fwhm(sigma_x))
#     assert_almost_equal(results.iloc[0][16], height / offset)
#     assert results.iloc[0][17] == 1.0
#     assert results.iloc[0][18] == 1.0
#     assert results.iloc[0][36] == "./data.tif_Bead_X20.5_Y20.2_Z20.0.png"
#
#     # Remove measured psf
#     widget.delete_measurement.click()
#     assert len(viewer.layers) == 2
#
#
# def test_psf_analysis_widget_advanced(make_napari_viewer, tmpdir, capsys):
#     viewer = make_napari_viewer()
#     height = 875
#     z, y, x = 20, 20, 20
#     sigma_z, sigma_y, sigma_x = 5, 3, 2
#     offset = 150
#
#     Z, Y, X = np.meshgrid(np.arange(40), np.arange(40), np.arange(40), indexing="ij")
#     gaussian = gaussian_3d(height, z, y, x, sigma_z, sigma_y, sigma_x, offset)
#
#     img = gaussian(Z, Y, X)
#
#     tmp_path = tmpdir.mkdir("test")
#     input_path = tmp_path.join("data.tif")
#     imwrite(input_path, img)
#     viewer.open(str(input_path))
#     viewer.add_points(np.array([[z, y, x]]))
#
#     widget = PsfAnalysis(viewer)
#     widget.xy_pixelsize.setValue(1)
#     widget.z_spacing.setValue(1)
#     widget.psf_box_size.setValue(30)
#
#     widget.temperature.setValue(25)
#     widget.airy_unit.setValue(1)
#     widget.bead_size.setValue(3)
#     widget.bead_supplier.setText("Beads by Dr. Gelman")
#     widget.mounting_medium.setText("Mountain Dew")
#     widget.objective_id.setText("no-ID")
#     widget.operator.setText("Tester")
#     widget.microscope_type.setText("Simulation")
#     widget.excitation.setValue(405)
#     widget.emission.setValue(568)
#     widget.comment.setText("I have a question.")
#
#     widget.microscope = QLineEdit("Microscope")
#     widget.extract_psfs.click()
#
#     captured = capsys.readouterr()
#     assert captured.out == ""
#
#     assert len(viewer.layers) == 3
#     assert viewer.layers[-1].data.shape == (1, 1000, 1000, 3)
#
#     widget.save_dir_line_edit.setText(str(tmp_path))
#     widget.save_button.click()
#
#     psf_img_path = glob(str(tmp_path.join("*.png")))[0]
#     assert psf_img_path == str(tmp_path.join("data.tif_Bead_X20.0_Y20.0_Z20.0.png"))
#
#     csv_path = glob(str(tmp_path.join("*.csv")))[0]
#     assert csv_path == str(
#         tmp_path.join(
#             "PSFMeasurement_{}_data.tif_Microscope_100_1.4.csv".format(
#                 datetime.today().strftime("%Y-%m-%d")
#             )
#         )
#     )
#
#     results = pd.read_csv(str(csv_path))
#     assert len(results.columns) == 48
#     assert results.columns[0] == "ImageName"
#     assert results.columns[1] == "Date"
#     assert results.columns[2] == "Microscope"
#     assert results.columns[3] == "Magnification"
#     assert results.columns[4] == "NA"
#     assert results.columns[5] == "Amplitude"
#     assert results.columns[6] == "Background"
#     assert results.columns[7] == "X"
#     assert results.columns[8] == "Y"
#     assert results.columns[9] == "Z"
#     assert results.columns[10] == "FWHM_X"
#     assert results.columns[11] == "FWHM_Y"
#     assert results.columns[12] == "FWHM_Z"
#     assert results.columns[13] == "PrincipalAxis_1"
#     assert results.columns[14] == "PrincipalAxis_2"
#     assert results.columns[15] == "PrincipalAxis_3"
#     assert results.columns[16] == "SignalToBG"
#     assert results.columns[17] == "XYpixelsize"
#     assert results.columns[18] == "Zspacing"
#     assert results.columns[19] == "cov_xx"
#     assert results.columns[20] == "cov_xy"
#     assert results.columns[21] == "cov_xz"
#     assert results.columns[22] == "cov_yy"
#     assert results.columns[23] == "cov_yz"
#     assert results.columns[24] == "cov_zz"
#     assert results.columns[25] == "sde_peak"
#     assert results.columns[26] == "sde_background"
#     assert results.columns[27] == "sde_X"
#     assert results.columns[28] == "sde_Y"
#     assert results.columns[29] == "sde_Z"
#     assert results.columns[30] == "sde_cov_xx"
#     assert results.columns[31] == "sde_cov_xy"
#     assert results.columns[32] == "sde_cov_xz"
#     assert results.columns[33] == "sde_cov_yy"
#     assert results.columns[34] == "sde_cov_yz"
#     assert results.columns[35] == "sde_cov_zz"
#     assert results.columns[36] == "Temperature"
#     assert results.columns[37] == "AiryUnit"
#     assert results.columns[38] == "BeadSize"
#     assert results.columns[39] == "BeadSupplier"
#     assert results.columns[40] == "MountingMedium"
#     assert results.columns[41] == "Objective_id"
#     assert results.columns[42] == "Operator"
#     assert results.columns[43] == "MicroscopeType"
#     assert results.columns[44] == "Excitation"
#     assert results.columns[45] == "Emission"
#     assert results.columns[46] == "Comment"
#     assert results.columns[47] == "PSF_path"
#
#     assert results.iloc[0][0] == "data.tif"
#     assert results.iloc[0][1] == datetime.today().strftime("%Y-%m-%d")
#     assert results.iloc[0][2] == "Microscope"
#     assert results.iloc[0][3] == 100.0
#     assert results.iloc[0][4] == 1.4
#     assert_almost_equal(results.iloc[0][5], height, decimal=0)
#     assert_almost_equal(results.iloc[0][6], offset, decimal=0)
#     assert_almost_equal(results.iloc[0][7], 20.0)
#     assert_almost_equal(results.iloc[0][8], 20.0)
#     assert_almost_equal(results.iloc[0][9], 20.0)
#
#     def fwhm(x):
#         return 2 * np.sqrt(2 * np.log(2)) * x
#
#     assert_almost_equal(results.iloc[0][10], fwhm(sigma_x))
#     assert_almost_equal(results.iloc[0][11], fwhm(sigma_y))
#     assert_almost_equal(results.iloc[0][12], fwhm(sigma_z))
#     assert_almost_equal(results.iloc[0][13], fwhm(sigma_z))
#     assert_almost_equal(results.iloc[0][14], fwhm(sigma_y))
#     assert_almost_equal(results.iloc[0][15], fwhm(sigma_x))
#     assert results.iloc[0][36] == 25
#     assert results.iloc[0][37] == 1
#     assert results.iloc[0][38] == 3
#     assert results.iloc[0][39] == "Beads by Dr. Gelman"
#     assert results.iloc[0][40] == "Mountain Dew"
#     assert results.iloc[0][41] == "no-ID"
#     assert results.iloc[0][42] == "Tester"
#     assert results.iloc[0][43] == "Simulation"
#     assert results.iloc[0][44] == 405
#     assert results.iloc[0][45] == 568
#     assert results.iloc[0][46] == "I have a question."
#     assert results.iloc[0][47] == "./data.tif_Bead_X20.0_Y20.0_Z20.0.png"
#
#
# def test_set_config(tmpdir):
#     config_dir = tmpdir.mkdir("config_dir")
#
#     save_dir = tmpdir.mkdir("save_dir")
#     settings = {"microscopes": ["Test1", "Test2"], "output_path": str(save_dir)}
#
#     config_name = config_dir.join("psf_analyze_settings.yaml")
#     with open(config_name, "w") as f:
#         yaml.dump(settings, f)
#
#     microscopes = get_microscopes(psf_settings_path=None)
#     assert microscopes == "Microscope"
#     microscopes = get_microscopes(psf_settings_path=str(config_name))
#     assert microscopes == ["Test1", "Test2"]
#
#     outpath = get_output_path(psf_settings_path=None)
#     assert outpath == pathlib.Path.home()
#     outpath = get_output_path(psf_settings_path=str(config_name))
#     assert str(outpath) == str(save_dir)
