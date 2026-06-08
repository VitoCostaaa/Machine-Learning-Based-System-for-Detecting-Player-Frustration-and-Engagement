# Development of a Machine Learning-Based System for Detecting Player Frustration and Engagement using In-Game Behavioural Data

## Overview

This bachelor's thesis presents the development of a machine learning-based system for detecting player frustration and engagement using in-game behavioral data. The objective is to design and implement a system that enables the automated recognition of emotional player states (frustration and engagement) based on measurable gameplay interactions.

A self-developed 2D game inspired by Flappy Bird was implemented as a controlled experimental environment. During gameplay, various behavioral data were collected and transformed into structured features. These features were used to train a machine learning model based on the Random Forest algorithm, enabling the identification of patterns associated with different player states.

In addition to the training process, the system includes a real-time analysis component that allows continuous classification of player behavior during gameplay. The results of the analysis were further visualized through graphical representations to provide an interpretable overview of the findings.

Overall, this thesis demonstrates that machine learning-based systems represent a promising approach for the detection and analysis of player behavior.

## Methodology

### Data Collection

The following behavioral features are extracted per game round:

* Duration of the round
* Score achieved
* Number of clicks
* Average and variance of click intervals
* Inter-trial interval (ITI)
* Panic rate (click frequency in the last seconds)
* Relative altitude at failure

The player manually labels each round after completion, providing ground truth for supervised learning.

For data acquisition, 8 test players participated in the experiment. Each participant completed approximately 40 game rounds, resulting in a balanced dataset of 320 recorded entries. This dataset forms the basis for model training and evaluation.

### Model Training

The collected dataset is used to train a machine learning model based on the Random Forest algorithm. The dataset is split into training and test sets to evaluate performance.

Key aspects:

* Supervised learning approach
* Feature-based classification
* Model persistence using serialization

### Real-Time Prediction

A trained model is used to perform real-time predictions of player states during gameplay. The system outputs both:

* Predicted class label
* Confidence score

### Data Visualization

Statistical plots are generated to analyze relationships between behavioral features and player states. These include:

* Boxplots (e.g., panic rate comparison)
* Scatterplots (e.g., performance vs. interaction patterns)

## Project Structure

```text
Project-Folder/
│
├── requirements.txt                 # Python dependencies
│
├── data/                            # Collected gameplay data
│   └── player_data.csv
│
├── models/                          # Trained machine learning models
│   └── detector_{player_name}.pkl
│
├── plots/                           # Generated visualizations
│   ├── comparison_panic_rate.png
│   ├── performance_analysis_{player_name}.png
│   └── feature_importance_{player_name}.png
│
├── game_environment/                # Game environment and data collection
│   └── game.py
│
├── machine_learning_system/         # Machine learning pipeline
│   ├── train_model.py               # Model training
│   ├── analyse_realtime.py          # Real-time prediction
│   └── visualise_results.py         # Data visualization
```

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Start Data Collection

```bash
python game.py
```

### 2. Train Model

```bash
python train_model.py
```

### 3. Run Real-Time Analysis

```bash
python analyse_realtime.py
```

### 4. Generate Visualizations

```bash
python visualise_results.py
```

## Requirements

* Python 3.11.9
* Pygame
* pandas
* NumPy
* scikit-learn
* joblib
* matplotlib
* seaborn

## Notes

The project uses a controlled experimental environment to ensure reproducibility and consistency of collected data.

Generated data, models and plots are excluded from version control.

## Author

Vito Marcel Costa


