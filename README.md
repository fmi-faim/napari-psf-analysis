# napari-psf-analysis

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI](https://img.shields.io/pypi/v/napari-psf-analysis.svg?color=green)](https://pypi.org/project/napari-psf-analysis)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-psf-analysis.svg?color=green)](https://python.org)
[![tests](https://github.com/fmi-faim/napari-psf-analysis/workflows/tests/badge.svg)](https://github.com/fmi-faim/napari-psf-analysis/actions)
[![codecov](https://codecov.io/gh/fmi-faim/napari-psf-analysis/branch/main/graph/badge.svg)](https://codecov.io/gh/fmi-faim/napari-psf-analysis)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-psf-analysis)](https://napari-hub.org/plugins/napari-psf-analysis)

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

---
![application_screenshot](figs/napari-psf-analysis_demo.gif)
<!-- start abstract -->
A plugin to analyse point spread funcitons (PSFs) of optical systems.
<!-- end abstract -->
## Usage
### Starting Point
To run a PSF analysis open an image of acquired beads. Add a point-layer
and indicate the beads you want to measure by adding a point.

### Run Analysis
Open the plugin (Plugins > napari-psf-analysis > PSF Analysis) and ensure
that your bead image and point layers are select in the `Basic` tab under
`Image` and `Points` respectively.
In the `Advanced` tab further information can be provided. Only the filled
in fields of the `Advanced` tab are saved in the output.

After verifying all input fields click `Extract PSFs`.

### Discard and Save Measurement
Once the PSF extraction has finished a new layer (`Analyzed Beads`) appears,
holding a summary
image for every selected bead.
Individual summaries can be discarded by clicking the `Delete Displayed
Measurement` button.

Results are saved to the selected `Save Dir` by clicking the `Save
Measurements` button.

Note: Beads for which the bounding box does not fit within the image are
automatically excluded from the analysis and no output is generated.


### Saved Data
Every image of the `Analyzed Beads` layer is saved as `{source_image_name}_X
{bead-centroid-x}_Y{bead-centroid-y}_Z{bead-centroid-z}.png` file.
Additionally a `PSFMeasurement_{source_image_acquisition_date}_
{source_image_name}_{microscope_name}_{magnification}_{NA}.csv` file is
stored containing the measured values and all filled in fields.

---
<!-- start install -->
## Installation
We recommend installation into a fresh conda environment.

### 1. Install napari
```shell
conda create -y -n psf-analysis -c conda-forge python=3.9

conda activate psf-analysis

conda install -c conda-forge napari pyqt
```

### 2. Install napari-aicsimageio and bioformats
Required if you want to open other files than `.tif` e.g. `.stk. `.

__Note:__ See [napari-aicsimageio](https://www.napari-hub.org/plugins/napari-aicsimageio) for more information about opening images.
```shell
conda install -c conda-forge openjdk bioformats_jar "aicsimageio[all]" napari-aicsimageio

conda deactivate
conda activate psf-analysis
```

### 3. Install napari-psf-analysis
You can install `napari-psf-analysis` via [pip]:

```shell
python -m pip install xmlschema
python -m pip install napari-psf-analysis
```

### 4. Optional `Set Config`
You can provide a config yaml file with the available microscope names and a default save directory.
This will change the `Microscope` text field to a drop down menu and change the default save directory.

`example_config.yaml`
```yaml
microscopes:
  - TIRF
  - Zeiss Z1
output_path: "D:\\psf_analysis\\measurements"
```

To use this config navigate to `Plugins > napari-psf-analysis > Set Config` and select the config file.

__Note:__ The save path is OS specific.

<!-- end install -->
### 5. Desktop Icon for Windows
Follow [these instructions](https://twitter.com/haesleinhuepf/status/1537030855843094529) by Robert Haase.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-psf-analysis" is free and open source software

## Issues

If you encounter any problems, please [file an issue](https://github.com/fmi-faim/napari-psf-analysis/issues) along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[@napari]: https://github.com/napari
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
