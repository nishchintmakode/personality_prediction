# Sentiment-Based Personality Profiling: Unveiling the Emotional Traits behind Textual Data

## Description

This repository contains the code and data for my dissertation titled "Sentiment-Based Personality Profiling: Unveiling the Emotional Traits behind Textual Data." The goal of this research is to explore the relationship between sentiment analysis and personality profiling, aiming to predict the Big Five personality traits (openness, conscientiousness, extraversion, agreeableness, and neuroticism) from textual data.

## Contents

1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Data Preprocessing](#data-preprocessing)
4. [Sentiment Analysis](#sentiment-analysis)
5. [Personality Detection](#personality-detection)
6. [Combined Model](#combined-model)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Usage](#usage)
9. [Requirements](#requirements)
10. [Contributing](#contributing)
11. [License](#license)

## Introduction

Sentiment analysis is a natural language processing technique commonly used to determine the sentiment (positive, negative, neutral) of textual data. This research goes beyond traditional sentiment analysis by investigating whether sentiment patterns in text can provide insights into an individual's underlying personality traits. The ultimate goal is to build a model that predicts Big Five personality traits based on sentiment analysis of textual data.

## Dataset Description

The research involves two main datasets:

1. **STS Dataset (Sentiment Analysis Training Dataset)**: A dataset containing 1.6 million text samples with associated sentiment polarity scores (0=negative, 2=neutral, 4=positive).

2. **myPersonality Dataset**: A dataset with textual data from various sources, including social media posts and personal blogs, along with Big Five personality trait scores.

Both datasets are stored in the `data` directory.

## Data Preprocessing

The initial step involves cleaning and preprocessing the datasets to prepare them for further analysis. The following preprocessing steps are performed:

- HTML parsing and special character removal.
- Handling hashtags, mentions, and retweets.
- Contraction expansion for text normalization.
- Stopword elimination.
- Text stemming and segmentation.

## Sentiment Analysis

To conduct sentiment analysis, the STS dataset is used to train a deep learning model. The sentiment analysis model is based on a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. After training the model, sentiment scores are assigned to the textual data in the myPersonality dataset using this trained model.

## Personality Detection

The myPersonality dataset is used for personality detection. The textual data, along with the assigned sentiment scores, are used to train another deep learning model. This model utilizes CNN and LSTM to predict the Big Five personality traits based on the text and sentiment features.

## Combined Model

The final model combines the sentiment analysis and personality detection steps. It takes textual data as input and performs sentiment analysis to assign sentiment scores. These scores, along with the raw text, are then used for predicting the Big Five personality traits using the trained personality detection model.

## Evaluation Metrics

The performance of the final model is evaluated using various metrics, including precision, recall, F1-score, and accuracy. These metrics provide insights into the model's ability to predict personality traits accurately.

## Usage

To use the code and reproduce the results, follow the steps outlined in the respective sections. Ensure the required datasets are present in the `data` directory. Additionally, make sure to have the necessary libraries and packages installed (listed in `requirements.txt`).

## Requirements

To run the code, you need the following libraries and packages:

- Python 3.x
- TensorFlow
- NLTK
- Pandas
- Numpy

Install the required packages using pip:
```bash
pip install -r requirements.txt
```
