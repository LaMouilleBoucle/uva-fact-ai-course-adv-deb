## FACT in AI, University of Amsterdam 2020

Fairness, Accountability, Confidentiality and Fairness in AI is a master's course at the UvA.
In the context thereof an attempt is made at reproducing a paper on fairness, namely
Mitigating Unwanted Biases with Adversarial Learning (Zhang et al., 2018).

This repository contains code for running the experiment from the paper on the UCI Adult dataset,
as well as extensions that implement the proposed adversarial network in order to debias the
UTKFace and UCI Communities and Crime datasets. To run the experiments,
please follow the instructions below.

### Prerequisites
Anaconda: https://www.anaconda.com/distribution/

### Getting Started
First open the Anaconda prompt, move to your desired working directory and clone this repository:
```bash
git clone https://github.com/LaMouilleBoucle/FACT2020.git
```

Then create and activate the environment necessary for running the experiments, using the following commands:
```bash
conda env create -f environment.yml
conda activate fact2020vlan
```

To deactivate the environment, use:
```bash
conda deactivate
```

% Here how to see results from the notebook

### Running the experiments

% Here an explanation on how to run experiments on each dataset yourself.
