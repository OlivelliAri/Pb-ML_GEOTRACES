### Mapping the global distribution of lead and its isotopes in seawater with explainable machine learning

#### Introduction
This GitHub repository contains the code developed for the article "Mapping the global distribution of lead and its isotopes in seawater with explainable machine learning" by Arianna Olivelli et al. (In review), as part of [her PhD research](https://profiles.imperial.ac.uk/a.olivelli21) at Imperial College London. 

The work has been funded by the Natural Environment Research Council (NE/S007415/1) through the Science and Solutions for a Changing Planet Doctoral Training Programme at the Grantham Institute - Climate Change and the Environment, Imperial College London. 

![alt text](https://github.com/OlivelliAri/Pb-ML_GEOTRACES/blob/main/docs/Pb_conc-cleanedSO_maps-no-coords_pcolormesh.jpg)

#### Usage
The code contained in this repository is designed for use in Jupyter Notebooks. To create a new environment, run the following commands in the terminal: 
```bash
conda env create -f environment.yml
conda activate DL_GEOTRACES
```

All notebooks with names starting with 'Pb-conc', 'Pb-67' and 'Pb87' refer to the Pb concentration, 206Pb/207Pb and 208Pb/207Pb models, respectively.

#### Data
The Pb concentration, 206Pb/207Pb and 208Pb/207Pb dataset created in ```'../Code/notebooks/Make_Pb_dataset.ipynb'``` and used for model development can be found at [https://doi.org/10.5281/zenodo.14261154](https://doi.org/10.5281/zenodo.14261154) as ```WOD_Pb_dataset-cleanedSOPbconc.csv```. 

Place it in the Data folder to run the Pb concentration, 206Pb/207Pb and 208Pb/207Pb notebooks located in ```'../Code/notebooks/*'```.

#### Licence 
The code in this repository is released under the MIT licence. 