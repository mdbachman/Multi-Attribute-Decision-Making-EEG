# Multi-Attribute-Decision-Making-EEG
Code repository for this code. 
Author: Matthew D. Bachman, matthewdbachman@gmail.com
Last update: July 9th, 2025

These scripts were originally run on a PC running Windows 11 Enterprise with an Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz and 64 GB of ram.
Necessary software dependencies can be found below. No non-standard hardware is required. The installation for each program should take no more than a few minutes.

## Behavioral Analyses ##
All of these analyses and related figures are contained in "BehavioralAnalyses.R"
The expected runtime for this script is no more than a minute or two.
Run using RStudio 2024.04.02+764.

## EEG Analyses ##
All analyses we run through Jupyter Notebook and Python with the following versions.
- conda            : 23.7.4
- python           : 3.11.5
- IPython          : 8.15.0
- ipykernel        : 6.25.0
- ipywidgets       : 8.0.4
- jupyter_client   : 7.4.9
- jupyter_core     : 5.3.0
- jupyter_server   : 1.23.4
- jupyterlab       : 3.6.3
- nbclient         : 0.5.13
- nbconvert        : 6.5.4
- nbformat         : 5.9.2
- notebook         : 6.5.4
- qtconsole        : 5.4.2
- traitlets        : 5.7.1

The following program/versions were also used in these scripts
- mne                           1.6.1
- matplotlib                    3.7.2
- matplotlib-inline             0.1.6
- numpy                         1.24.3
- pandas                        2.0.3
- scipy                         1.11.1
- statsmodels                   0.14.0

Here is the layout and organization of the scripts. Please note that you will likely need to change file/folder pathways to match your system.
### EEG Preprocessing
- All takes place within "preprocessingEEGData.ipynb". This output is used for all future EEG analyses. Each participant may take 10-60 minutes or so to process, depending on your comfort with EEG processing.

### Temporally-Specific Decoders
- The head script is: TemporallySpecificAnalyses_runScript.py
- From here it references: TemporallySpecificAnalyses_SVC_mainFunctions.py, with timebin = 10 and conditions set to "All". This script may take 30-60 minutes per participant.
- This data is collated for statistical analysis and plotting in: TemporallySpecificAnalyses_SVC_plotResults.ipynb  This script should take only a few minutes to run.

### Temporal Generalization Analyses
- From here it references: TemporalGeneralizationAnalyses_runScript.py
- The head script is:  TemporalGeneralizationAnalyses_mainFunctions.py.  This script may take 30-60 minutes per participant.
- This data is collated for statistical analysis and plotting in: TemporalGeneralizationAnalyses_plotResults.ipynb  This script should take only a few minutes to run.
  
### Spatial Analyses
- Also relies on the same data used for "temporally-specific decoders"
- All analyses conducted in: SpatialAnalyses_plotResults.py. This script may take 5-10 minutes to run in its entirety.

### Value-boosted Analyses
- The head script is: TemporallySpecificAnalyses_runScript.py
- From here it references: TemporallySpecificAnalyses_SVC_mainFunctions_byBlock, with timebin = 20, and conditions set to "Faces" and "Colors". This script may take 30-60 minutes per participant.
- This data is then processed and exported in: ValueBoostedAnalyses_run_and_plot_results.ipynb. The permutation testing may take a day or two to process. The remaining analyses should take less than 15 minutes to fully run.
  
### Supplemental Analyses
- Exemplar specific topos: SpatialAnalyses_plotResults.ipynb.  This script should take only a few minutes to run.
- Support vector regressions:
  - The head script is: TemporallySpecificAnalyses_runScript.py
  - From here it references: TemporallySpecificAnalyses_SVR_mainFunctions.py. This script may take 60-120 minutes to run per participant.
  - This data is collated for statistical analysis and plotting in: TemporallySpecificAnalyses_SVR_plotResults.ipynb. This script should take only a few minutes to run.
- Value-boosted analysis by reward magnitude: ValueBoostedAnalyses_run_and_plot_results.ipynb.  This script should take only a few minutes to run, after permutation testing is complete.
  
### Auxillary scripts with custom functions
- EEG_auxiliary_module_sptm_wICA.py
- EEG_auxiliary_module_guggenmos.py
