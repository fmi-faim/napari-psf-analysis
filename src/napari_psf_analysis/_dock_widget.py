import pathlib
from datetime import datetime
from os.path import basename, dirname, exists, getctime, join

import napari.layers
import numpy as np
import pkg_resources
import yaml
from magicgui import magic_factory
from napari import viewer
from napari._qt.qthreading import FunctionWorker, thread_worker
from napari.settings import get_settings
from napari.utils.notifications import show_info
from qtpy.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from skimage.io import imsave

from napari_psf_analysis.psf_analysis.psf_analysis import (
    analyze_bead,
    build_summary_figure,
    create_result_table,
    localize_beads,
    merge,
)
from napari_psf_analysis.psf_analysis.utils import fwhm


def get_microscopes(psf_settings_path):
    if psf_settings_path and exists(psf_settings_path):
        settings = load_settings(psf_settings_path)

        if settings and "microscopes" in settings.keys():
            return [s for s in settings["microscopes"]]

    return "Microscope"


def get_output_path(psf_settings_path):
    if psf_settings_path and exists(psf_settings_path):
        settings = load_settings(psf_settings_path)

        if settings and "output_path" in settings.keys():
            return pathlib.Path(settings["output_path"])

    return pathlib.Path.home()


def load_settings(psf_settings_path):
    with open(psf_settings_path) as stream:
        settings = yaml.safe_load(stream)
    return settings


def get_psf_analysis_settings_path():
    config_pointer = join(
        dirname(get_settings()._config_path), "psf_analysis_config_pointer.yaml"
    )
    if exists(config_pointer):
        settings = load_settings(config_pointer)
        if settings and "psf_analysis_config_file" in settings.keys():
            return settings["psf_analysis_config_file"]

    return None


def layer_widget(layer: napari.layers.Image):
    return layer


class PsfAnalysis(QWidget):
    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent=parent)
        self._viewer: viewer = napari_viewer
        napari_viewer.layers.events.inserted.connect(self._layer_inserted)
        napari_viewer.layers.events.removed.connect(self._layer_removed)
        napari_viewer.layers.selection.events.changed.connect(self._on_selection)

        self.bead_imgs = None
        self.results = None

        self.setLayout(QVBoxLayout())
        self.setMinimumWidth(300)
        self.setMaximumHeight(500)

        setting_tabs = QTabWidget(parent=self)
        setting_tabs.setLayout(QHBoxLayout())

        basic_settings = QWidget(parent=setting_tabs)
        advanced_settings = QWidget(parent=setting_tabs)

        setting_tabs.addTab(basic_settings, "Basic")
        setting_tabs.addTab(advanced_settings, "Advanced")
        advanced_settings.setLayout(QFormLayout())
        basic_settings.setLayout(QFormLayout())

        self.cbox_img = QComboBox(parent=basic_settings)
        self.cbox_point = QComboBox(parent=basic_settings)

        basic_settings.layout().addRow(QLabel("Image", basic_settings), self.cbox_img)
        basic_settings.layout().addRow(
            QLabel("Points", basic_settings), self.cbox_point
        )

        self.date = QDateEdit(datetime.today())
        basic_settings.layout().addRow(
            QLabel("Acquisition Date", basic_settings), self.date
        )

        microscope_options = get_microscopes(get_psf_analysis_settings_path())
        if isinstance(microscope_options, list):
            self.microscope = QComboBox(parent=basic_settings)
            self.microscope.addItems(microscope_options)
        else:
            self.microscope = QLineEdit(microscope_options)
        basic_settings.layout().addRow(
            QLabel("Microscope", basic_settings), self.microscope
        )

        self.magnification = QSpinBox(parent=basic_settings)
        self.magnification.setMinimum(0)
        self.magnification.setMaximum(10000)
        self.magnification.setValue(100)
        self.magnification.setSingleStep(10)
        basic_settings.layout().addRow(
            QLabel("Magnification", basic_settings), self.magnification
        )

        self.objective_id = QLineEdit("obj_1")
        basic_settings.layout().addRow(
            QLabel("Objective ID", basic_settings), self.objective_id
        )

        self.na = QDoubleSpinBox(parent=basic_settings)
        self.na.setMinimum(0.0)
        self.na.setMaximum(1.7)
        self.na.setSingleStep(0.05)
        self.na.setValue(1.4)
        basic_settings.layout().addRow(QLabel("NA", basic_settings), self.na)

        self.xy_pixelsize = QDoubleSpinBox(parent=basic_settings)
        self.xy_pixelsize.setMinimum(0.0)
        self.xy_pixelsize.setMaximum(10000.0)
        self.xy_pixelsize.setSingleStep(10.0)
        self.xy_pixelsize.setValue(65.0)
        basic_settings.layout().addRow(
            QLabel("XY-Pixelsize [nm]", basic_settings), self.xy_pixelsize
        )

        self.z_spacing = QDoubleSpinBox(parent=basic_settings)
        self.z_spacing.setMinimum(0.0)
        self.z_spacing.setMaximum(10000.0)
        self.z_spacing.setSingleStep(10.0)
        self.z_spacing.setValue(200.0)
        basic_settings.layout().addRow(
            QLabel("Z-Spacing [nm]", basic_settings), self.z_spacing
        )

        self.psf_yx_box_size = QDoubleSpinBox(parent=basic_settings)
        self.psf_yx_box_size.setMinimum(1.0)
        self.psf_yx_box_size.setMaximum(1000000.0)
        self.psf_yx_box_size.setSingleStep(500.0)
        self.psf_yx_box_size.setValue(2000.0)
        basic_settings.layout().addRow(
            QLabel("PSF YX Box Size [nm]", basic_settings), self.psf_yx_box_size
        )

        self.psf_z_box_size = QDoubleSpinBox(parent=basic_settings)
        self.psf_z_box_size.setMinimum(1.0)
        self.psf_z_box_size.setMaximum(1000000.0)
        self.psf_z_box_size.setSingleStep(500.0)
        self.psf_z_box_size.setValue(6000.0)
        basic_settings.layout().addRow(
            QLabel("PSF Z Box Size [nm]", basic_settings), self.psf_z_box_size
        )

        self.temperature = QDoubleSpinBox(parent=advanced_settings)
        self.temperature.setMinimum(-100)
        self.temperature.setMaximum(200)
        self.temperature.setSingleStep(0.1)
        self.temperature.clear()
        advanced_settings.layout().addRow(
            QLabel("Temperature", advanced_settings), self.temperature
        )

        self.airy_unit = QDoubleSpinBox(parent=advanced_settings)
        self.airy_unit.setMinimum(0)
        self.airy_unit.setMaximum(1000)
        self.airy_unit.setSingleStep(0.1)
        self.airy_unit.clear()
        advanced_settings.layout().addRow(
            QLabel("Airy Unit", advanced_settings), self.airy_unit
        )

        self.bead_size = QDoubleSpinBox(parent=advanced_settings)
        self.bead_size.setMinimum(0)
        self.bead_size.setMaximum(1000)
        self.bead_size.setSingleStep(1)
        self.bead_size.clear()
        advanced_settings.layout().addRow(
            QLabel("Bead Size [nm]", advanced_settings), self.bead_size
        )

        self.bead_supplier = QLineEdit()
        advanced_settings.layout().addRow(
            QLabel("Bead Supplier", advanced_settings), self.bead_supplier
        )

        self.mounting_medium = QLineEdit()
        advanced_settings.layout().addRow(
            QLabel("Mounting Medium", advanced_settings), self.mounting_medium
        )

        self.operator = QLineEdit()
        advanced_settings.layout().addRow(
            QLabel("Operator", advanced_settings), self.operator
        )

        self.microscope_type = QLineEdit()
        advanced_settings.layout().addRow(
            QLabel("Mircoscope Type", advanced_settings), self.microscope_type
        )

        self.excitation = QDoubleSpinBox(parent=advanced_settings)
        self.excitation.setMinimum(0)
        self.excitation.setMaximum(1000)
        self.excitation.setSingleStep(1)
        self.excitation.clear()
        advanced_settings.layout().addRow(
            QLabel("Excitation", advanced_settings), self.excitation
        )

        self.emission = QDoubleSpinBox(parent=advanced_settings)
        self.emission.setMinimum(0)
        self.emission.setMaximum(1000)
        self.emission.setSingleStep(1)
        self.emission.clear()
        advanced_settings.layout().addRow(
            QLabel("Emission", advanced_settings), self.emission
        )

        self.comment = QLineEdit()
        advanced_settings.layout().addRow(
            QLabel("Comment", advanced_settings), self.comment
        )

        self.layout().addWidget(setting_tabs)

        self.extract_psfs = QPushButton("Extract PSFs")
        self.extract_psfs.clicked.connect(lambda: self.prepare_measure())
        self.layout().addWidget(self.extract_psfs)

        self.delete_measurement = QPushButton("Delete Displayed Measurement")
        self.delete_measurement.setEnabled(False)
        self.delete_measurement.clicked.connect(self.delete_measurements)
        self.layout().addWidget(self.delete_measurement)

        dir_selection_dialog = QWidget(parent=self)
        dir_selection_dialog.setLayout(QHBoxLayout())
        self.save_path = QFileDialog()
        self.save_path.setFileMode(QFileDialog.DirectoryOnly)
        self.save_path.setDirectory(
            str(get_output_path(get_psf_analysis_settings_path()))
        )

        self.save_dir_line_edit = QLineEdit()
        self.save_dir_line_edit.setText(self.save_path.directory().path())

        choose_dir = QPushButton("...")
        choose_dir.clicked.connect(self.select_save_dir)

        dir_selection_dialog.layout().addWidget(
            QLabel("Save Dir", dir_selection_dialog)
        )
        dir_selection_dialog.layout().addWidget(self.save_dir_line_edit)
        dir_selection_dialog.layout().addWidget(choose_dir)

        self.layout().addWidget(dir_selection_dialog)
        self.save_button = QPushButton("Save Measurements")
        self.save_button.setEnabled(True)
        self.save_button.clicked.connect(self.save_measurements)
        self.layout().addWidget(self.save_button)

        self.current_img_index = -1
        self.cbox_img.currentIndexChanged.connect(self._img_selection_changed)
        self.fill_layer_boxes()

    def select_save_dir(self):
        self.save_path.exec_()
        self.save_dir_line_edit.setText(self.save_path.directory().path())

    def fill_layer_boxes(self):
        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self.cbox_img.addItem(str(layer))
                self._img_selection_changed()
            elif isinstance(layer, napari.layers.Points):
                self.cbox_point.addItem(str(layer))

    def _img_selection_changed(self):
        if self.current_img_index != self.cbox_img.currentIndex():
            self.current_img_index = self.cbox_img.currentIndex()
            for layer in self._viewer.layers:
                if str(layer) == self.cbox_img.itemText(self.cbox_img.currentIndex()):
                    self.date.setDate(
                        datetime.fromtimestamp(getctime(layer.source.path))
                    )

    def _layer_inserted(self, event):
        if isinstance(event.value, napari.layers.Image):
            self.cbox_img.insertItem(self.cbox_img.count() + 1, str(event.value))
        elif isinstance(event.value, napari.layers.Points):
            self.cbox_point.insertItem(self.cbox_point.count() + 1, str(event.value))

    def _layer_removed(self, event):
        if isinstance(event.value, napari.layers.Image):
            items = [self.cbox_img.itemText(i) for i in range(self.cbox_img.count())]
            self.cbox_img.removeItem(items.index(str(event.value)))
            self.changed_manually = False
            self.current_img_index = -1
            self._img_selection_changed()
        elif isinstance(event.value, napari.layers.Points):
            items = [
                self.cbox_point.itemText(i) for i in range(self.cbox_point.count())
            ]
            self.cbox_point.removeItem(items.index(str(event.value)))

    def _on_selection(self, event):
        if self._viewer.layers.selection.active is not None:
            self.delete_measurement.setEnabled(
                self._viewer.layers.selection.active.name == "Analyzed Beads"
            )

    def prepare_measure(self):
        if isinstance(self.microscope, QComboBox):
            microscope = self.microscope.currentText()
        else:
            microscope = self.microscope.text()

        magnification = self.magnification.value()
        na = self.na.value()
        z_spacing = self.z_spacing.value()
        xy_pixelsize = self.xy_pixelsize.value()
        psf_box_size_yx = self.psf_yx_box_size.value()
        psf_box_size_z = self.psf_z_box_size.value()
        img_layer = None
        for layer in self._viewer.layers:
            if str(layer) == self.cbox_img.currentText():
                img_layer = layer
        if img_layer is None:
            return

        point_layer = None
        for layer in self._viewer.layers:
            if str(layer) == self.cbox_point.currentText():
                point_layer = layer
        if point_layer is None:
            return

        if len(img_layer.data.shape) != 3:
            raise NotImplementedError(
                f"Only 3 dimensional data is "
                f"supported. Your data has {(img_layer.data.shape)} dimensions."
            )

        from bfio.bfio import NapariReader

        if isinstance(img_layer.data, NapariReader):
            img_data = np.transpose(img_layer.data.br.read(), [2, 0, 1]).copy()
        else:
            img_data = img_layer.data.copy()

        point_data = point_layer.data.copy()
        name = basename(img_layer.source.path)

        def _on_done(result):
            if result is not None:
                measurement_stack, measurement_scale = result
                self._viewer.add_image(
                    measurement_stack,
                    name="Analyzed Beads",
                    interpolation="bicubic",
                    rgb=True,
                    scale=measurement_scale,
                )
                self._viewer.dims.set_point(0, 0)
                self._viewer.reset_view()
            self.setEnabled(True)

        worker: FunctionWorker = self.measure(
            microscope,
            magnification,
            na,
            z_spacing,
            xy_pixelsize,
            psf_box_size_z,
            psf_box_size_yx,
            name,
            img_data,
            point_data,
        )

        worker.returned.connect(_on_done)
        worker.start()

        self.setEnabled(False)

    @thread_worker(progress={"total": 0})
    def measure(
        self,
        microscope,
        magnification,
        na,
        z_spacing,
        xy_pixelsize,
        psf_box_size_z,
        psf_box_size_yx,
        name,
        img_data,
        point_data,
    ):
        date = datetime(*self.date.date().getDate()).strftime("%Y-%m-%d")
        version = pkg_resources.get_distribution("napari_psf_analysis").version
        spacing = (z_spacing, xy_pixelsize, xy_pixelsize)
        patch_size = (psf_box_size_z, psf_box_size_yx, psf_box_size_yx)

        beads, offsets = localize_beads(
            img=img_data, points=point_data, spacing=spacing, patch_size=patch_size
        )

        accumulated_results = None
        self.bead_imgs = {}
        for bead, offset in zip(beads, offsets):
            res = analyze_bead(bead=bead, spacing=spacing)

            summary_fig = build_summary_figure(
                bead_img=bead,
                spacing=spacing,
                location=(res["z_mu"], res["y_mu"], res["x_mu"]),
                fwhm_values=(res["z_fwhm"], res["y_fwhm"], res["x_fwhm"]),
                cov_matrix_3d=np.array(
                    [
                        [res["zyx_cxx"], res["zyx_cyx"], res["zyx_czx"]],
                        [res["zyx_cyx"], res["zyx_cyy"], res["zyx_czy"]],
                        [res["zyx_czx"], res["zyx_czy"], res["zyx_czz"]],
                    ]
                ),
                date=date,
                version=version,
            )
            yx_cov_matrix = np.array(
                [[res["yx_cyy"], res["yx_cyx"]], [res["yx_cyx"], res["yx_cxx"]]]
            )
            yx_pc = np.sort(np.sqrt(np.linalg.eigvals(yx_cov_matrix)))[::-1]

            res["z_mu"] += offset[0] * spacing[0]
            res["y_mu"] += offset[1] * spacing[1]
            res["x_mu"] += offset[2] * spacing[2]
            res["zyx_z_mu"] += offset[0] * spacing[0]
            res["zyx_y_mu"] += offset[1] * spacing[1]
            res["zyx_x_mu"] += offset[2] * spacing[2]
            res["yx_pc1_fwhm"] = fwhm(yx_pc[0])
            res["yx_pc2_fwhm"] = fwhm(yx_pc[1])
            res["image_name"] = name
            res["date"] = date
            res["microscope"] = microscope
            res["mag"] = magnification
            res["NA"] = na
            res["yx_spacing"] = spacing[1]
            res["z_spacing"] = spacing[0]
            res["version"] = version
            accumulated_results = merge(accumulated_results, res)

            centroid = np.round([res["x_mu"], res["y_mu"], res["z_mu"]], 1)
            bead_name = "{}_Bead_X{}_Y{}_Z{}".format(res["image_name"], *centroid)
            self.bead_imgs[bead_name] = summary_fig

        self.results = create_result_table(results=accumulated_results)

        if len(self.bead_imgs) > 0:
            measurement_stack = np.stack(
                [self.bead_imgs[k] for k in self.bead_imgs.keys()]
            )
            bead_scale = self._viewer.layers[self.cbox_img.currentText()].scale
            bead_shape = self._viewer.layers[self.cbox_img.currentText()].data.shape

            measurement_scale = np.array(
                [
                    bead_scale[0],
                    bead_scale[1] / measurement_stack.shape[1] * bead_shape[1],
                    bead_scale[1] / measurement_stack.shape[1] * bead_shape[1],
                ]
            )
            return measurement_stack, measurement_scale

    def delete_measurements(self):
        idx = self._viewer.dims.current_step[0]
        self.results = self.results.drop(idx).reset_index(drop=True)
        tmp = {}
        for i, k in enumerate(self.bead_imgs.keys()):
            if i != idx:
                tmp[k] = self.bead_imgs[k]

        self.bead_imgs = tmp
        if len(self.bead_imgs) == 0:
            self._viewer.layers.remove_selected()
        else:
            self._viewer.layers.selection.active.data = np.stack(
                [self.bead_imgs[k] for k in self.bead_imgs.keys()]
            )

    def save_measurements(self):
        outpath = self.save_dir_line_edit.text()
        bead_paths = []
        for k in self.bead_imgs.keys():
            bead_paths.append(join("./", k + ".png"))
            imsave(join(outpath, k + ".png"), self.bead_imgs[k])

        if self.temperature.text() != "":
            self.results["Temperature"] = self.temperature.value()

        if self.airy_unit.text() != "":
            self.results["AiryUnit"] = self.airy_unit.value()

        if self.bead_size.text() != "":
            self.results["BeadSize"] = self.bead_size.value()

        if self.bead_supplier.text() != "":
            self.results["BeadSupplier"] = self.bead_supplier.text()

        if self.mounting_medium.text() != "":
            self.results["MountingMedium"] = self.mounting_medium.text()

        if self.objective_id.text() != "":
            self.results["Objective_id"] = self.objective_id.text()

        if self.operator.text() != "":
            self.results["Operator"] = self.operator.text()

        if self.microscope_type.text() != "":
            self.results["MicroscopeType"] = self.microscope_type.text()

        if self.excitation.text() != "":
            self.results["Excitation"] = self.excitation.value()

        if self.emission.text() != "":
            self.results["Emission"] = self.emission.value()

        if self.comment.text() != "":
            self.results["Comment"] = self.comment.text()

        self.results["PSF_path"] = bead_paths
        entry = self.results.iloc[0]
        self.results.to_csv(
            join(
                outpath,
                "PSFMeasurement_"
                + entry["Date"]
                + "_"
                + entry["ImageName"]
                + "_"
                + entry["Microscope"]
                + "_"
                + str(entry["Magnification"])
                + "_"
                + str(entry["NA"])
                + ".csv",
            ),
            index=False,
        )

        show_info("Saved results.")


@magic_factory(
    config_path={"label": "Config Path", "filter": "*.yaml"},
    call_button="Use config file.",
)
def set_config(config_path=pathlib.Path.home()):
    if exists(config_path):
        settings = load_settings(config_path)
        if "microscopes" in settings.keys() or "output_path" in settings.keys():
            config_pointer_path = join(
                dirname(get_settings()._config_path), "psf_analysis_config_pointer.yaml"
            )

            with open(config_pointer_path, "w") as yamlfile:
                conf_dict = {"psf_analysis_config_file": str(config_path)}
                yaml.safe_dump(conf_dict, yamlfile)
