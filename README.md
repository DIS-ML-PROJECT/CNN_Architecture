# DIS22 ML Project

## Team CNN Architecture
This Repository contains the code from the DIS22 ML Project Team CNN Architecture.
Most of the Code is adapted from the original code repository of [sustainlab-group](https://github.com/sustainlab-group/africa_poverty).

Please find a Sphinx Documentation in this repository, which for now contains only the code of the CNN Architecture Team.

For a visualised dataflow, look at the miro board [Timeflow CNN](https://miro.com/app/board/o9J_lC6R8PY=/).
While the diagrams "DHS Data sustainlab-group" and "LSMS Data" show the original dataflow by sustainlab-group, the diagram "DHS Data DIS 22" shows the shortened and adapted way used in this repository. 

While the sustainlab-group dealed with DHS and LSMS data, the DIS22 ML Project deals only with DHS data

## Running this Repository
1. run the notebook `data_analysis/dhs_create_pkl.ipynb`
2. run script `train_directly.py`
3. run script `extract_features_dhs.py`
4. run notebook `models/dhs_ridge_resnet.ipynb`

For summaries of these scripts and notebooks please check out the corresponding files (.md for .py and in-line documentation for .ipynb files).
