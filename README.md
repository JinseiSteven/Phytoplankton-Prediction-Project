# Coupling Abiotic and Biotic datasets - jam.rs

Machine learning and data analysis project with the main goal being to couple abiotic, phytoplankton and irradiance datasets, after which analysis and prediction of phytoplankton concentration can be done. This is a project conducted by students from the University of Amsterdam.

This project is commisioned by Dr. Louis Peperzak, on behalf of the NIOZ for a second year Bachelor AI project.


# Files

The repository is structured in the following way:
 - `data`
    - `ABIO` - Dataset containing recordings of abiotic variables in different sampling locations of the north sea.
    - `PHYTO` - Dataset containing recordings of phytoplankton concentrations in different sampling locations of the north sea.
    - `Irradiance` - Dataset containing recordings of partial irradiance recordings in the north sea.
    - `MERGED_DATA_INTERPOLATED` - Merged dataset containing both abiotic and phytoplankton recordings, with added calculations of extra variables.
 - `scripts`
    - `clustering` - Files containing python scripts detailing our process for clustering the sampling locations and phytoplankton types.
    - `data_analysis` - Data analysis for combining the different datasets.
    - `neural_network` - Different files containing our neural network architecture and applications of it on the phytoplankton data.
    - `plots` - File containing different plotted graphs and maps with our analysis results.
    - `rdaclust` - Files describing the process of applying RDA on the dataset.
    - `time_series_forecasting` - Different applications of time series forecasting on the abiotic dataset.
 - `miscellaneous` - Folder containing useful literature and pdfs for the project.
 - `rscripts` - Folder containing multiple data analysis scripts in R, focussed on Redundancy Analysis.


# Authors

- Stephan Visser (13977571)
- Rijk van der Meer (13986554)
- Milan Tool (14599996)
- Ardjano Mark (14713926)
- Joost Weerheim (13769758)
