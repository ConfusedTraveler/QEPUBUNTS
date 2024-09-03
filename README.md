# QEPUBUNTS
Implementation of our SIGKDD 2024 Paper "Quantifying and Estimating the Predictability Upper Bound of Univariate Numeric Time Series"

# Datasets

Will be added in the next few days.

# Research Paper Experiments

This repository contains the Python code used to perform the experiments described in our research paper. The experiments are organized into separate folders, with scripts to generate figures and results, which are stored in the `Results` folder under each experiement.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Figure 2](#figure-1)
- [Experiment 1](#experiment-1)
- [Experiment 2](#experiment-2)
- [Experiment 3](#experiment-3)
- [Utils](#utils)
- [Results](#results)

## Overview

The code is organized as follows:

- `challenges/`: Contains code to generate Figure 2.
- `Experiments/`: Contains subdirectories for each experiment:
  - `Exp1/`: Scripts for Experiment 1.
  - `Exp2/`: Scripts for Experiment 2.
  - `Exp3/`: Scripts for Experiment 3 (To be detailed later).

## Requirements

1. Clone the repository:
    ```bash
    # Clone the repository to your local machine
    git clone https://github.com/JamalSZ/QEPUBUNTS.git

    # Change directory to the cloned repository
    cd QEPUBUNTS
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Figure 2

To generate Figure 2, follow these steps:

1. Navigate to the `Challenges` directory:
    ```bash
    cd challenges
    ```

2. Run the script:
	```bash
	   # N=number of states and n length 
	   # python fig2.py first N n
	   python fig2.py 1 50 10000
    ```

    ```bash
   	# After first run
    python fig2.py 0
    ```

The output will be saved in the `Challenges/Results/` folder.

## Experiment 1

To reproduce the results for Experiment 1, follow these steps:

1. Navigate to the `Exp1` directory:
    ```bash
    cd Experiments/Exp1
    ```

2. Run the following scripts in the order given:
    ```bash
    python pimax.py
    python ar1.py
    python markov_predict.py
    ```

3. Finally, generate the plots by running:
    ```bash
    python plot_exp1.py
    ```

All outputs will be saved in the `Exp1/Results/` directory.

## Experiment 2

To reproduce the results for Experiment 2, follow these steps:

1. Navigate to the `Exp2` directory:
    ```bash
    cd Experiments/Exp2
    ```

2. Run the following scripts:
    ```bash
    python exp2_fig4a.py
    python exp2_fig4b.py
    ```

3. Finally, generate the combined plot by running:
    ```bash
    python plot_exp2_fig4a_4b.py
    ```

All outputs will be saved in the `Exp2/Results/` directory.


## Utils: Entropy Estimators

The `Utils` folder contains scripts that implement entropy estimators and the predictability upper bound calculations used across the experiments. These scripts include:

- `LZ1.py`: Implements the NLZ1 entropy estimator.
- `LZ2.py`: Implements the NLZ2 entropy estimator.
- `gen_data.py`: Generates synthetic Markov Series.
- `ComputeH_exact.py`: Computes the exact entropy rate of know (Ideal Scenario) Markov Series.
- `Compute_PIMAX.py`: Computes the predictability upper bound based on entropy estimates.

These scripts are utilized by the other experiment scripts as needed. You can also run these scripts independently to compute entropy estimates and predictability bounds for your own data.
## Results

The results for all experiments and figures are saved in the `Results` directory.

