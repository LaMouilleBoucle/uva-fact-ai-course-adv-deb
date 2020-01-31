## FACT in AI, University of Amsterdam 2020

Fairness, Accountability, Confidentiality and Fairness in AI is a master's course at the UvA.
In the context thereof an attempt is made at reproducing a paper on fairness, namely
Mitigating Unwanted Biases with Adversarial Learning ([Zhang et al., 2018](https://arxiv.org/abs/1801.07593)).

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

To view the notebook with our experimental results, run:
```bash
jupyter notebook data_analysis.ipynb
```

### Running the experiments
New experiments can be conducted using the main.py file. The dataset to be used can be specified by passing it as an argument on the command line. Other arguments are:

```bash
OUTPUT FOR THE ARGS
```

Not all of these are relevant or appropriate for the different datasets. Below you will find commands for running an experiment on a specific dataset, including arguments specific to it, with the default settings.

#### UCI Adult dataset
The UCI adult dataset is the default data for doing the debiasing experiments. To train and test without debiasing, run: 
```bash
python main.py 
```

To debias, run:
```bash
python main.py --debias
```

To replicate the results presented in the notebook, run:
```bash
python main.py --batch_size 128 --predictor_lr 0.1 --n_epochs 10
python main.py --debias --batch_size 128 --predictor_lr 0.01 --adversary_lr 0.001 --n_epochs 30
```

#### UCI Communities and Crime dataset
To train, validate and test on the UCI Communities and Crime dataset, run:

```bash
python main.py --dataset crime --val
```

To replicate the results presented in the notebook, run:
```bash
python main.py --dataset crime --n_epochs 50
python main.py --dataset crime --debias --batch_size 64 --predictor_lr 0.002 --adversary_lr 0.005 --n_epochs 210
```

#### UTKFace dataset
The UTKFace set is not present in the data folder of this repository. The data is downloaded and placed into the right local folder when experimenting with it for the first time. To train, validate and test on the UTK Face dataset, run:

```bash
python main.py --dataset face --val
```

To replicate the results presented in the notebook, run:
```bash
python main.py --dataset face --batch_size 128 --predictor_lr 0.001 --n_epochs 30
python main.py --dataset face --batch_size 128 --predictor_lr 0.001 --adversary_lr 0.001 --n_epochs 30
```

### Authors
- Vanessa Botha - [*VanessaBotha*](https://github.com/VanessaBotha)
- Nithin Holla - [*Nithin-Holla*](https://github.com/Nithin-Holla)
- Azamat Omuraliev - [*AzamatOmu*](https://github.com/AzamatOmu)
- Leila Talha - [*LaMouilleBoucle*](https://github.com/LaMouilleBoucle)

### Acknowledgements
We would like to express great appreciation to IBM for releasing the [AI Fairness 360 toolkit](https://github.com/IBM/AIF360) that has been of inspiration to us, when parameter settings required to reproduce results were not mentioned by Zhang and colleagues. In addition we are grateful for the datasets made publicly available by UCI and [susanqq](https://github.com/susanqq). Finally we would like thank Leon Lang for providing us with advice and feedback and for swiftly responding to our e-mails, answering our questios.

\\ Include IBM team here, as they were a source of inspiration?
\\ UCI/UTK for the datasets or better as reference?
