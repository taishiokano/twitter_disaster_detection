# Best Classifiers for Classifying Real Disaster Tweets from Fake Disaster Tweets

This is the repository for CAPP30255 - Advanced Machine Learning for Public Policy Project

## Brief Explanation
We work on improving machine learning models to classify real disasters on Twitter using natural language processing techniques. Identifying real disasters on Twitter is crucial for local governments to manage resources efficiently and respond quickly. Using Twitter as a tool can be helpful, but distinguishing real disasters from fake ones is difficult. We construct models for this purpose using a dataset that comes from a Kaggle competition titled “Natural Language Processing with Disaster Tweets”, combined with “Disasters on Social Media”.

## Folders and Files

- `data`: persists all data files
    - `disasters-on-social-media.csv` and `nlp-with-disaster-tweets-train.csv` are the raw text data files downloaded from Kaggle
    - `cleaned-train-tweets.csv` is the main data file that used in our project. (See the process of cleaning data in `01-preprocessing.ipynb`)
- Python notebooks: each notebook illustrates detail of one pipeline / module. We also provides the reference of each notebook to each section in the report
    - `01-preprocessing.ipynb`: preprocessing train dataset (section 4)
    - `02-data-exploration.ipynb`: doing the data exploration (section 5)
    - `03-logistic-regression-with-bow-cbow`: modeling the baseline machine learning model (logistic regression) with BoW / GloVe / fastText in PyTorch (section 6.2)
    - `04-lstm-with-context-free-embeddings.ipynb`: modeling deep learning models (LSTM and BiLSTM) with BoW / GloVe / fastText (section 7.1)
    - `05-lstm-trained-from-scratch.ipynb`: modeling BiLSTM model with an embedding matrix trainable from sctach and updatable with the context (section 7.2)
    - `11-baseline-models.ipynb`: Implementation of Naive Bayes and Logistic Regression with Bug of Words and tf-idf using Scikit-Learn. (section 6)
    - `12-DistilBERT.ipynb`: Implementation of BERT / DistilBERT using Hugging Face (section 7)
- The others
    - `install.sh`: Make environment to run the following python file and some of notebooks.
    - `search_tweets.py`: Getting test data via Twitter API
    - `product_last_year/utils.py`: codes extracted from out last year's project and referenced

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
