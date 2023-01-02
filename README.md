# Cognitive Cascades -- Media Ecosystem Model

## Author: Nick Rabb 

#### Contact: nicholas.rabb@tufts.edu

<hr/>

## Project Overview

TO BE FILLED IN

## Experiment & Results Replication

### Using Netlogo Behaviorspace

To run simulations and generate polarization results, open the `cog-cascades-media-ecosystem.nlogo` file in NetLogo, and go to *Tools -> BehaviorSpace*. One experiment is called *conditions-to-polarization_cognitive*, and is the one we used for results in Rabb & Cowen, 2022. For our paper results, we used the following parameter settings:

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

### Using data from Rabb & Cowen 2022

#### Analysis of polarization data

To prepare your Python environment for doing this analysis, you must first import the `data.py` file:

```py
from data import *
```

If you want to use full simulation result data it must either be requested from the authors (email above), or generated with NetLogo's BehaviorSpace. To run simulations and generate polarization data, refer to the section above.

Then, to get the data into a usable dataframe that aggregates results over simulation trials, run the following code:

```py
multidata = get_conditions_to_polarize_multidata(<YOUR_PATH>)
polarization_data = polarization_analysis(multidata)
```

Alternatively, we provide the dataframe results of running `polarization_analysis()` in the `data` directory in this GitHub repository. Simply use `pandas.load_csv()` to load the data frame. If this is the method you use, then loading both `polarizing-df.csv` and `nonpolarizing-df.csv` into a dictionary called `polarization_data` as follows will allow the rest of the README instructions to work:

```py
polarized_df = pd.read_csv('data/polarizing-df.csv')
nonpolarized_df = pd.read_csv('data/nonpolarizing-df.csv')
polarization_data = { 'polarizing': polarized_df, 'nonpolarizing': nonpolarized_df }
```

`polarization_data` will then be dictionary containing several `pandas` dataframes which you can use to see which results were classified as polarized and nonpolarized. These dataframes can be queried to analyze results of the simulations. Importantly, the dataframes contain the *means* of parameter combinations' results -- since there were 5 simulation trials per parameter combination, each resultant polarization measure is based off of the mean polarization values across 5 trials. The dataframe can be accessed and analyzed as follows:

```py
polarized_df = polarization_data['polarizing']
print(polarized_df.columns)
>> Index(['translate', 'tactic', 'media_dist', 'citizen_dist', 'epsilon',
       'graph_type', 'ba-m', 'repetition', 'lr-intercept', 'lr-slope', 'var',
       'start', 'end', 'delta', 'max', 'polarized'],
      dtype='object')

print(polarized_df.iloc[0])
>> translate                     0
>> tactic          broadcast-brain
>> media_dist              uniform
>> citizen_dist             normal
>> epsilon                       0
>> graph_type      barabasi-albert
>> ba-m                          3
>> repetition                    1
>> lr-intercept            2.99003
>> lr-slope               0.025486
>> var                    0.181005
>> start                  3.025278
>> end                    4.071444
>> delta                  1.046166
>> max                    4.193444
>> polarized                     1
>> Name: 3, dtype: object
```

Trends in the data can be analyzed through a variety of methods, and we will explain the ones we used below.

#### Analysis of effect of fragmentation & exposure parameters (Table 3)

Code that we used for this analysis is contained in the `polarization_results_by_fragmentation_exposure(polarization_data)` function in `data.py`. It finds points in the dataset that match certain values of `epsilon`, `gamma (translate)` and `h_G (homophily)`. One command that can be used to query the dataset in that manner is:

```py
results = polarized_df.query("epsilon==0 and translate==1 and graph_type=='barabasi-albert'")
```

This set of rows can be used to determine what portion polarized and what portion failed to polarize. This is the basis of our analyses.

#### Other analyses of effects in Tables 4 & 5

The same techinque as above is utilized to generate the results contained in Tables 4 and 5. The following functions can be run to replicate those results:

```py
polarization_results_by_tactic_exposure(polarization_data) # Table 4
polarization_results_by_broadcast_distributions(polarization_data) # Table 5
```

#### Logistic regressions (Section 5 -- Results)

Our logistic regressions were performed with the function `logistic_regression_polarization(polarization_data)` in `data.py`. This was used to determine effects on polarization given certain simulation parameters.

#### Generating visual charts

To generate charts based off of the experiment results, you should use the `process_polarizing_conditions_cognitive()` function, giving it the path of the downloaded data. This function has several variables within it that specify which parameter values to process. For example, if you wanted to generate diagrams for the cases where *gamma* (beta function translation factor) was 0, 1, and 2, you would set `cognitive_translate = [0,1,2]`.

**Note: The parameter values that you give the analysis function must have a corresponding directory where that data is stored from the NetLogo simulation. In other words, you must have gathered results for those parameter values. The parameter values that are contained in our results set are listed in Tables 1 and 2 in the paper.**