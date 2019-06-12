# cs229-project
# Classifying Tweets Based on Climate Change Stance

*Authors: Jason Qin and Chuma Kabaghe*

## Data acquisition and preprocessing
Data acquisition code is located in:
* ./code/twitterdata.py : collects tweets with the following hashtags - "globalwarminghoax", "globalwarmingisahoax", "climatechange", "climatehustle", "climatechangefraud"

Preprocessing removes punctuation, regularizes capitalization, and organizes data such that it can be read in by downstream tools to construct models.

Preprocessing code is located in:
* ./code/preprocess_tweets.py

## Modelling and Analysis
Modeling includes testing different 1) models, 2) hyperparameters, 3) downsampling extents

Relevant code in:
* ./code/util.py : helper functions for reading in labeled and unlabeled data, converting tweets to word frequency matrices, plotting functionality
* ./code/NBclassifier.py : code for running unigram MNB, bigram MNB, and MNB-EM
* ./code/multiclass_qns3vm.py : code for running S3VM model
* ./code/downsample.py : code for downsampling data, and finding prediction accuracy on train/val/test data
* ./code/analyze_downsampled_data.ipynb : code for analyzing and plotting downsampling data

## Data
Relevant data used for modeling are in the ./data directory
* ./data/2016_train.csv : labeled training data
* ./data/2016_test.csv : labeled test data
* ./data/2016_val.csv : labeled validation data

* ./data/unlabelled3_06.txt : preprocessed tweets (collected from TweePy and then preprocessed)