**IN PROGRESS**

# Best Classifiers for Classifying Real Disaster Tweets from Fake Disaster Tweets
## Brief Explanation
We work on improving machine learning models to classify real disasters on Twitter using natural language processing techniques. Identifying real disasters on Twitter is crucial for local governments to manage resources efficiently and respond quickly. Using Twitter as a tool can be helpful, but distinguishing real disasters from fake ones is difficult. We construct models for this purpose using a dataset that comes from a Kaggle competition titled “Natural Language Processing with Disaster Tweets”, combined with “Disasters on Social Media”.


## Updated on 05/02/2023

- Please use `data/cleaned-train-tweets.csv` for doing further task. The change is I join bugged list of string to string
- To cope with this change, I modify `11-baseline-models.ipynb`.
- I also add my work. `03-logistic-regression-with-bow-cbow.ipynb`. The logistic regression with bow and cbow representations, and using PyTorch. You can copy codes for implementing new PyTorchModel 
- I will finish write up soon! (Please remind me to fix your wordcloud, due to change in dataset)


## How to use
### Setting up
``` sh
bash install.sh
source env/bin/activate
```

## Dataset
Please use `data/cleaned-train-tweets.csv` for doing further task.
We documented things such as code for cleaning and preprocessing data in `01-preprocessing.ipynb`



### Search tweets
``` py
python search_tweets.py
```
