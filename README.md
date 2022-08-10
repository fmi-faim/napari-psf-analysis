# napari-psf-analysis

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI](https://img.shields.io/pypi/v/napari-psf-analysis.svg?color=green)](https://pypi.org/project/napari-psf-analysis)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-psf-analysis.svg?color=green)](https://python.org)
[![tests](https://github.com/fmi-faim/napari-psf-analysis/workflows/tests/badge.svg)](https://github.com/fmi-faim/napari-psf-analysis/actions)
[![codecov](https://codecov.io/gh/fmi-faim/napari-psf-analysis/branch/main/graph/badge.svg)](https://codecov.io/gh/fmi-faim/napari-psf-analysis)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-psf-analysis)](https://napari-hub.org/plugins/napari-psf-analysis)

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

---
![application_screenshot](./figs/napari-psf-analysis_demo.gif)
A plugin to analyse point spread funcitons (PSFs) of optical systems.

## Usage
### Starting Point
To run a PSF analysis open an image of acquired beads. Add a point-layer
and indicate the beads you want to measure by adding a point.

### Run Analyis
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

For the demo gif above the following table is saved:

|ImageName               |Date      |Microscope|Magnification|NA |Amplitude        |Background        |X                 |Y                 |Z                 |FWHM_X            |FWHM_Y            |FWHM_Z           |PrincipalAxis_1  |PrincipalAxis_2   |PrincipalAxis_3   |SignalToBG        |XYpixelsize|Zspacing|cov_xx            |cov_xy             |cov_xz           |cov_yy           |cov_yz             |cov_zz           |sde_peak         |sde_background      |sde_X              |sde_Y              |sde_Z              |sde_cov_xx        |sde_cov_xy        |sde_cov_xz        |sde_cov_yy        |sde_cov_yz        |sde_cov_zz        |Comment        |PSF_path                                                   |
|------------------------|----------|----------|-------------|---|-----------------|------------------|------------------|------------------|------------------|------------------|------------------|-----------------|-----------------|------------------|------------------|------------------|-----------|--------|------------------|-------------------|-----------------|-----------------|-------------------|-----------------|-----------------|--------------------|-------------------|-------------------|-------------------|------------------|------------------|------------------|------------------|------------------|------------------|---------------|-----------------------------------------------------------|
|100x_1_conf488Virtex.tif|2022-03-03|Microscope|100          |1.4|5169.285944782688|110.18242108419038|2605.150064016795 |2778.5159415714847|2223.0135754293333|205.83948141718292|193.26935441526453|673.333737589546 |673.8395142125338|204.76810910270055|192.64368202154003|46.91570482765882 |65.0       |200.0   |7640.8541537792735|-13.739986414927806|2461.133326066555|6736.131301493249|-1761.0316747094355|81760.83213128973|6.07230627881135 |0.0428908418118399  |0.0918331172590858 |0.08622431079805634|0.30039707647545805|16.058330709118493|10.658894965304015|37.31418836385325 |14.157198228507017|34.96459026793594 |171.82884173452643|This is a demo.|./100x_1_conf488Virtex.tif_Bead_X2605.2_Y2778.5_Z2223.0.png|
|100x_1_conf488Virtex.tif|2022-03-03|Microscope|100          |1.4|6135.157449356215|110.45693133711426|2579.1750636343136|2665.971138363623 |2236.556334604507 |210.3414510686511 |197.70468562397645|688.7450949822045|689.0569862986882|209.38893767869146|197.62898832950918|55.543435573379554|65.0       |200.0   |7978.739450747766 |71.03860432658904  |2445.186895806542|7048.853370163676|194.33624544039762 |85546.37080807924|6.427131015205848|0.04702336845023734 |0.08368900397680192|0.0786607859867236 |0.2740328722683267 |14.954667965040326|9.937027929319177 |34.76778883464614 |13.21218589548314 |32.53756104188376 |160.34176367808521|This is a demo.|./100x_1_conf488Virtex.tif_Bead_X2579.2_Y2666.0_Z2236.6.png|
|100x_1_conf488Virtex.tif|2022-03-03|Microscope|100          |1.4|5622.117168996411|110.12846686628077|2387.1094900635194|2575.7229681487956|2280.3111520343996|214.6603075814513 |210.29252188934618|707.5199298039764|708.9044265632655|211.44593966381632|208.88171792203815|51.05053515203157 |65.0       |200.0   |8309.75169186637  |216.78540934275736 |4866.991328053463|7975.027887810296|2320.8118890686897 |90273.83813806984|6.840381484837519|0.052450225692372655|0.09919251952617283|0.09717397041964153|0.32694856225371205|18.089302104598307|12.53155240411209 |42.809428898592365|17.360668540008014|41.444943537461285|196.52752653343444|This is a demo.|./100x_1_conf488Virtex.tif_Bead_X2387.1_Y2575.7_Z2280.3.png|

With the three summary images:

![summaries](figs/summaries.png)

---

## Installation
We recommend installation into a fresh conda environment.

### 1. Install napari
```shell
conda create -y -n psf-analysis -c conda-forge python=3.9

conda activate psf-analysis

python -m pip install "napari[all]"
```

### 2. Install napari-aicsimageio and bioformats
Required if you want to open other files than `.tif` e.g. `.stk. `.

__Note:__ See [napari-aicsimageio](https://www.napari-hub.org/plugins/napari-aicsimageio) for more information about opening images.
```shell
conda install -c conda-forge openjdk

conda deactivate
conda activate psf-analysis

python -m pip install "bfio[bioformats]"
python -m pip install "aicsimageio[all]"
python -m pip install napari-aicsimageio
```

### 3. Install napari-psf-analysis
You can install `napari-psf-analysis` via [pip]:

```shell
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
