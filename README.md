# Best Classifiers for Classifying Real Disaster Tweets from Fake Disaster Tweets
## Brief Explanation
We work on improving machine learning models to classify real disasters on Twitter using natural language processing techniques. Identifying real disasters on Twitter is crucial for local governments to manage resources efficiently and respond quickly. Using Twitter as a tool can be helpful, but distinguishing real disasters from fake ones is difficult. We construct models for this purpose using a dataset that comes from a Kaggle competition titled “Natural Language Processing with Disaster Tweets”, combined with “Disasters on Social Media”.

## Explanation of each file
- `01-preprocessing.ipynb` (49 lines)
- `02-data-exploration.ipynb` (79 lines)
- `03-logistic-regression-with-bow-cbow.ipynb` (287 lines)
- `04-lstm-with-context-free-embeddings.ipynb` (390 lines)
- `05-lstm-trained-from-scratch.ipynb` (356 lines)
- `11-baseline-models.ipynb` (172 lines): Implementation of Naive Bayes and Logistic Regression with Bug of Words and tf-idf using Scikit-Learn.
- `12-BERT.ipynb` (716 lines): Implementation of BERT / DistilBERT using Hugging Face
- `install.sh` (34 lines): Make environment to run the following python file and some of notebooks.
- `search_tweets.py` (46 lines): Getting test data via Twitter API
- `product_last_year/utils.py` (196 lines): codes extracted from out last year's project and referenced

## List files containing code not written by a team member, or not written as part of this project 
- `product_last_year/utils.py`


## Other relevant infomation
### Setting up
Run the following code.
``` sh
bash install.sh
source env/bin/activate
```

### Search tweets
Run the following code after setting up as in the above commands.
``` py
python search_tweets.py
```

### Dataset
Please use `data/cleaned-train-tweets.csv` for doing further task. We documented things such as code for cleaning and preprocessing data in `01-preprocessing.ipynb`
