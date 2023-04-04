# Cognitive Cascades -- Media Ecosystem Model

## Author: Nick Rabb 

#### Contact: nicholas.rabb@tufts.edu

<hr/>

## Project Overview

TO BE FILLED IN

## Software Prerequisites

### Python

Our software uses Python 3.8 with several packages installed. Please install the following packages:

- numpy
- scipy
- sklearn
- networkx
- pandas
- matplotlib

### NetLogo

The NetLogo version we use is 6.1.1. Please find and download that version from their official website at https://ccl.northwestern.edu/netlogo/6.1.1/.

## Experiment & Results Replication

### Static Media Ecosystem Model

#### Using Netlogo Behaviorspace

To run simulations and generate polarization results, open the `cog-cascades-trust.nlogo` file in NetLogo, and go to *Tools -> BehaviorSpace*. One experiment is called *conditions-to-polarization_cognitive*, and is the one we used for results in Rabb, Cowen, and de Ruiter 2023 for the static media ecosystem model experiments. For our paper results, we used the following parameter settings:

```
["belief-resolution" 7]
["tick-end" 100]
["brain-type" "discrete"]
["N" 100]
["contagion-on?" false]
["cognitive-fn" "sigmoid-stubborn"]
["spread-type" "cognitive"]
["cognitive-scalar" 20]
["cognitive-exponent" 4]
["cognitive-translate" 0 1 2]
["institution-tactic" "broadcast-brain" "appeal-mean" "appeal-median"]
["media-ecosystem" "distribution"]
["media-ecosystem-dist" "uniform" "normal" "polarized"]
["media-dist-normal-mean" 3]
["media-dist-normal-std" 1]
["media-ecosystem-n" 20]
["message-repeats" 1]
["citizen-init-dist" "uniform" "normal" "polarized"]
["cit-init-normal-mean" 3]
["cit-init-normal-std" 1]
["epsilon" 0 1 2]
["graph-type" "barabasi-albert" "ba-homophilic"]
["ba-m" 3]
["repetition" [0 1 2]]
```

We also set `Repetitions` to 5 and `Time Limit` to 100.

Running this experiment will generate directories with simulation results in the following directory naming scheme: `ROOT/cognitive-translate/institution-tactic/media-ecosystem-dist/citizen-init-dist/epsilon/graph-type/ba-m/repetition`. Each unique parameter combination will output 5 simulation results to that directory. Our data analysis is set to read directories in this fashion and aggregate results based off of these directory structures. If the structure of directories for output is changed, then the analysis functions will need to be changed as well.

#### Analysis of polarization data

To prepare your Python environment for doing this analysis, you must first import the `data_analysis.py` file:

```py
from data_analysis import *
```

If you want to use full simulation result data it must either be requested from the authors (email above), or generated with NetLogo's BehaviorSpace. To run simulations and generate polarization data, refer to the section above. Otherwise, there are data files in XYZ repository, that were used for results in the paper, that can be loaded from `.csv`.

If you are analyzing results that you created using the BehaviorSpace extension, then run the following code to get data from the simulation output:

```py
multidata = get_static_sweep_multidata(<PATH_TO_YOUR_DATA>)
df = multidata_as_dataframe(multidata, ['translate','tactic','media_dist','citizen_dist','epsilon','graph_type','ba_m','repetition'])
df.to_csv(<PATH_TO_YOUR_DATA_OUTPUT>)
```

Then, once you've either saved your own data as a `.csv` or obtained access to our data as a `.csv`, you can run the analyses below to recreate our reuslts. The easiest way to run the entire battery of analyses is to run:

```py
static_model_total_analysis(<DIRECTORY_TO_PUT_DATA>,<PATH_TO_YOUR_CSV>,['translate','tactic','media_dist','citizen_dist','epsilon','graph_type','ba_m','repetition'])
```

Data files will then be generated in the directory you specified in the first argument. To perform a more custom analysis of polarization data, you can use the following few lines, taken from `static_total_model_analysis()`:


```py
polarization_slope = 0.01
polarization_intercept = 5.5
polarization_data = polarization_analysis(multidata, polarization_slope, polarization_intercept)
```

The result will have several dataframes contained within one dictionary. To analyze results from every simulation trial, not averaged across trials on the same network topology and parameter combination, use `polarization_data['polarization_all_df']`. If you want to analyze the data reduced to the mean time series across simulation runs for unique parameter and topology combinations, use `polarization_data['polarization_df']`.

#### Logistic regressions (Section 5 -- Results)

Our logistic regressions were performed with the function `logistic_regression_polarization(polarization_data)` in `data.py`. This was used to determine effects on polarization given certain simulation parameters.

### Dynamic Media Ecosystem Model

The analysis process for dynamic media ecosystem model results is the same as above for the static model. First, data needs to be in a `.csv` file for all of the simulation runs. Whether that's accomplished by getting a data file from the authors, or by running your own experiments, the process is the same afterward.

If you wanted to run either of the two experiments that we ran in the paper, they are contained in the BehaviorSpace tool in this project's NetLogo file. The main experiment is broken up into 3 experiment files called `parameter_sweep_low_res-translate-X` where X ranges from [0,2]. The other experiment is called `parameter_sweep_low_res_low-media`. This supports our results testing a low availability media ecosystem.

If you need to process the results of those experiments before doing any analysis, use either the function `get_low_res_sweep_multidata(<YOUR_PATH>)` or `get_low_res_low_media_multidata(<YOUR_PATH>)`. Each will return you a multidata object, which you can convert to a dataframe and write to `.csv` as follows:

```py
multidata = get_low_res_sweep_multidata(<PATH_TO_YOUR_DATA>)
# The parameter for column names here is only for low_res_param_sweep, for low_res_low_media, use: ['translate','media_dist','tactic','citizen_dist','zeta_citizen','zeta_media','citizen_memory_len','repetition']
df = multidata_as_dataframe(multidata, ['translate','tactic','media_dist','media_n','citizen_dist','zeta_citizen','zeta_media','citizen_memory_len','repetition'])
df.to_csv(<PATH_TO_YOUR_DATA_OUTPUT>)
```

Then, once you have your respective data in a `.csv` file, you can recreate all our analyses by simply running either `low_res_sweep_total_analysis(<OUTPUT_DIR>,<YOUR_DATA_FILE>)` or `low_res_low_media_total_analysis(<OUTPUT_DIR>,<YOUR_DATA_FILE>)`. This will output data files, LaTeX formatted files, and chart images for all the analyses we performed.


#### Generating visual charts

To generate charts based off of the experiment results, you should use the `process_polarizing_conditions_cognitive()` function, giving it the path of the downloaded data. This function has several variables within it that specify which parameter values to process. For example, if you wanted to generate diagrams for the cases where *gamma* (beta function translation factor) was 0, 1, and 2, you would set `cognitive_translate = [0,1,2]`.

**Note: The parameter values that you give the analysis function must have a corresponding directory where that data is stored from the NetLogo simulation. In other words, you must have gathered results for those parameter values. The parameter values that are contained in our results set are listed in Tables 1 and 2 in the paper.**