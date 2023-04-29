**IN PROGRESS**

# Best Classifiers for Classifying Real Disaster Tweets from Fake Disaster Tweets
## Brief Explanation
We work on improving machine learning models to classify real disasters on Twitter using natural language processing techniques. Identifying real disasters on Twitter is crucial for local governments to manage resources efficiently and respond quickly. Using Twitter as a tool can be helpful, but distinguishing real disasters from fake ones is difficult. We construct models for this purpose using a dataset that comes from a Kaggle competition titled “Natural Language Processing with Disaster Tweets”, combined with “Disasters on Social Media”.

## How to use
### Setting up
``` sh
bash install.sh
source env/bin/activate
```

## Dataset
Please use `data/cleaned-tokenized-train-tweets.csv` for doing further task.
We documented things such as code for cleaning and preprocessing data in `01-preprocessing.ipynb`



### Search tweets
``` py
python search_tweets.py
```
