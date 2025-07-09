# Multi-Attribute-Decision-Making-EEG
Code repository for this code. 
Author: Matthew D. Bachman, matthewdbachman@gmail.com
Last update: July 9th, 2025

## Behavioral Analyses ##
All of these analyses and related figures are contained in "BehavioralAnalyses.R"
Run using RStudio 2024.04.02+764.

## EEG Analyses ##
All analyses we run through Jupyter Notebook and Python with the following versions.
conda            : 23.7.4
python           : 3.11.5
IPython          : 8.15.0
ipykernel        : 6.25.0
ipywidgets       : 8.0.4
jupyter_client   : 7.4.9
jupyter_core     : 5.3.0
jupyter_server   : 1.23.4
jupyterlab       : 3.6.3
nbclient         : 0.5.13
nbconvert        : 6.5.4
nbformat         : 5.9.2
notebook         : 6.5.4
qtconsole        : 5.4.2
traitlets        : 5.7.1

The following program/versions were also used in these scripts
mne                           1.6.1
matplotlib                    3.7.2
matplotlib-inline             0.1.6
numpy                         1.24.3
pandas                        2.0.3
scipy                         1.11.1
statsmodels                   0.14.0

Here is the layout and organiztion of the scripts. Please note that you will likely need to change file/folder pathways to match your system.
### EEG Preprocessing
- All takes place within "preprocessingEEGData.ipynb". This output is used for all future EEG analyses.

### Temporally-Specific Decoders
- The head script is: TemporallySpecificAnalyses_runScript.py
- From here it references: TemporallySpecificAnalyses_SVC_mainFunctions.py, with timebin = 10 and conditions set to "All"
- This data is collated for statistical analysis and plotting in: TemporallySpecificAnalyses_SVC_plotResults.ipynb

### Temporal Generalizaiton Analyses
- From here it references: TemporalGeneralizationAnalyses_runScript.py
- The head script is:  TemporalGeneralizationAnalyses_mainFunctions.py
- This data is collated for statistical analysis and plotting in: TemporalGeneralizationAnalyses_plotResults.ipynb
- 
### Spatial Analyses
- Also relies on the same data used for "temporally-specific decoders"
- All analyses conducted in: SpatialAnalyses_plotResults.py

### Value-boosted Analyses
- The head script is: TemporallySpecificAnalyses_runScript.py
- From here it references: TemporallySpecificAnalyses_SVC_mainFunctions_byBlock, with timebin = 20, and conditions set to "Faces" and "Colors".
- This data is then processed and exported in: ValueBoostedAnalyses_run_and_plot_results.ipynb
- 
### Supplemental Analyses
- Exemplar specific topos: SpatialAnalyses_plotResults.ipynb
- Support vector regressions:
  - The head script is: TemporallySpecificAnalyses_runScript.py
  - From here it references: TemporallySpecificAnalyses_SVR_mainFunctions.py
  - This data is collated for statistical analysis and plotting in: TemporallySpecificAnalyses_SVR_plotResults.ipynb
- Value-boosted analysis by reward magnitude: ValueBoostedAnalyses_run_and_plot_results.ipynb
- 
### Auxillary scripts with custom functions
- EEG_auxiliary_module_sptm_wICA.py
- EEG_auxiliary_module_guggenmos.py
