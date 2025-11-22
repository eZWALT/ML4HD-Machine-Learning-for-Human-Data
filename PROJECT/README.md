# ML4HD Project: Motor Imagery Classification for Brain Computer Interface 

<h1 align="center">
<img src="resources/EEG_example.png" width="512">
</h1>


## Description

This project analyzes over 5 GB of EEG data to develop a robust classifier for Motor Imagery (MI). Our goal is threefold: to provide the user with a practical insight on the latest trends, techniques and surveys related to the modelling, preprocessing, and representation of EEG data, to build a high-accuracy predictive model for Brain-Computer Interfaces (BCI), and to serve as a practical tutorial and best-practice guide for handling complex neurophysiological data.

### Classification Tasks
We address three progressively challenging tasks, aiming for high accuracy across all scales of motor representation:

| Task | Classes | Description | Difficulty |
| :--- | :--- | :--- | :--- |
| **I** | 3 | Left Hand, Right Hand, Passive | Standard |
| **II** | 6 | Hand/Leg/Tongue movements & Passive | Advanced |
| **III** | 5 | **Individual Five Fingers** movement | **Challenging** |


## Contents

The directory is structured in the following way:

- **src**: Contains the notebooks used throughout the experimentation and can be used as 
- **resources**: Contains images and miscellaneous documents
- **documents**: Final report and report template 

## TODO

Our current roadmap is:

- Pre-processing techniques
    1. Compression? 
    2. Clustering?
    3. Autoencoding?
    4. ???
- Data representations (classic feature extraction, spectral representation, etc …) 
    1. Classic Feature Extraction
    2. Spectral Representations
    3. ???
- Different NN architectures (novel, or combinations of existing ones)
    1. CNNs?
    2. GNNs?
    3. SNNs? (Prof. Rossi said its a bad approach)

## Usage 

### 0. Virtual Environment 

> WARNING: Its recommended to create a virtual environment  (venv / pyenv or conda) to ensure full reproducibility, using python 3.12.11 and the fixed requirements inside of requirements.txt file. 

In order to create a virtual environment using conda:

```sh
#Create environment:
conda env create -f environment.yml
#activate:
conda activate machine-learning-env
#check:
python src/utils/env_check.py
```

To create a virtual environment using pyenv:

```sh
# Install python 3.12.11
pyenv install 3.11.13
# Create the virtual environment ML4HD
pyenv virtualenv 3.11.13 ml4hd
# Activate it 
pyenv activate
# Make it the default environment for this project
pyenv local ml4hd 
# Install the dependencies using pip
pip install -r requirements.txt
```

### 1. Executing the Pipelines 

In order to inspect and execute the different solutions and data pipelines that have been built for this project, refer to the main notebooks inside of main **src/** directory, also refer to the drive collab versions for more GPU compute power:

1. CNNs solution
2. GNNs solution
3. SNNs solution

**WORK IN PROGRESS**

