# TCGA_stats: Statistics for TCGA-BRCA
This repository is used to generate the statistics for TCGA BRCA. ```run.sh``` contains all the necessary code to run this pipeline and generate figures. THe relevant paths need to be provided to ensure that the figures are saved in the relevant places.

## Clean-up
We use the original patient and case info obtained from GDC to generate the stats. These are nested json files, and generating statistics from them is hard. Therefore we use the ```clean_json.py``` file to convert this json file to a dataFrame to allow us to work with it and wrangle it easily.

## Generate Statistics
We use the cleaned and sorted dataFrame to get our statistics and other information using the ```get_summaries.py``` file.
