# Secondary Dominant Detection

## Setup
1. Install Fluidsynth.

    The Fluidsynth cli tool is required to generate wave files from midi files.
    Download and install Fluidsynth from https://www.fluidsynth.org/. Ensure that 
    Fluidsynth is added to your system PATH so that it can be accessed from the command line.
    Installation can be validated with the following command:
    ```bash
    fluidsynth -h
    ```

2. Install Python 3.7 or higher.
3. In the root directory, set up your python virtual environment by doing the following
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage
To run the program, run:
```bash
    python main.py
```

The program has the following command line arguments:

- `--feature-type [type]`: Specifies the feature representation used as input to the model
This can be any of the following:
    - `global-mfcc`: Uses globally averaged MFCCs.
    - `per-chord-mfcc`: Uses concatenated per-chord averaged MFCCs.
    - `global-tonnetz`: Uses globally averaged Tonnetz features.
    - `per-chord-tonnetz`: Uses concatenated per-chord averaged Tonnetz features.
    - `hpcp`: Uses concatenated per-chord averaged HPCP features.
    - `hpcp-tonnetz`: Uses concatenated per-chord averaged HPCP and Tonnetz features combined.

- `--model-type [type]`: Specifies a specific machine learning model to use for secondary dominant detection.
This can be any of the following:
    - `logistic-regression`: Uses a Logistic Regression model.
    - `svm`: Uses a Support Vector Machine model.

- `--gen-songs [num_songs]`: Generates a specified number of synthetic chord progressions for training and testing.
If this argument is provided, the program will forcefully regenerate the [num_songs] amount of synthetic songs even if they already exist.
If num_songs is not provided, its value will default to 200.

- `--regen-features`: If this flag is provided, the program will re-extract features from the audio files even if they already exist.

## Dataset
The dataset used in this project is synthetically generated using a custom chord progression generator. The generator creates a
`data` directory in the root of the project. This directory contains the following subdirectories: `diatonic` and `non-diatonic`.
The `diatonic` directory contains chord progressions that only use diatonic chords, while the `non-diatonic` directory contains 
chord progressions that include secondary dominants.

## Graphs
Previously generated graphs can be found in the `graphs` directory in the root of the project. As the project is run again,
new graphs will be generated and saved in this directory, overwriting any existing graphs with the same name.

## Main Experiment
The main experiment reported in the paper can be reproduced by running the following commands:
```bash
    python main.py --feature-type hpcp-tonnetz --gen-songs 2000 --model-type svm
    python main.py --feature-type hpcp-tonnetz --model-type logistic-regression
```

Graphs are opened automatically after the experiment is complete and accuracy results are printed to the console.
Typical runtime for the full experiment is around 30 minutes on a standard laptop as the program needs to generate  
and analyze around 13 GB worth of synthetic audio data.