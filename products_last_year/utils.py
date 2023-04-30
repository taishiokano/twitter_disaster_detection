import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict
from wordcloud import WordCloud
from PIL import Image
import os

"""
How to use this module
1. Make your prediction. My module accepts a y_predict array like this:
array([0, 0, 0, ..., 0, 1, 0], dtype=int64)

2. Make classification report based on your prediction. We can compare accuracy, 
precision, recall and f1-score.

3. Convert your prediction into 4 classification by classifying_results.
This function allows you to get 4 types of tweets: 
 tweets classified as disaster-related correctly(actual = 1, predict = 1),
 tweets classified as nondisaster-related correctly(actual = 0, predict = 0),
 tweets misclassified as nondisaster-related(actual = 1, predict = 0),
 tweets misclassified as disaster-related(actual = 0, predict = 1)

4. Make bar charts of ngrams for misclassified tweets by analysing_ngrams.

5. Make word clouds for misclassified tweets by word_cloud. 

A sample usage is here:

from utils import analysing_data as ad
1.y_predict = model.predict(X_test)
2.classification_report(y_test, y_predict)
3.true_positive, true_negative, false_negative, false_positive
     = ad.classifying_results(y_test, y_predict, train)
4.ad.analysing_ngrams(false_negative, false_positive, 2, N=10)
5.ad.word_cloud(false_negative, max_words=100)
"""

def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(' ') if token != '' and not token in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]

def create_corpus_df(tweet):
    corpus=[]
    for x in tweet['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus

STOPWORDS = ['amp', 'where', 'done', 'if', 'before', 'll', 'very', 'keep', 'something', 'nothing', 'thereupon', 
        'may', 'why', '’s', 'therefore', 'you', 'with', 'towards', 'make', 'really', 'few', 'former', 
        'during', 'mine', 'do', 'would', 'of', 'off', 'six', 'yourself', 'becoming', 'through', 
        'seeming', 'hence', 'us', 'anywhere', 'regarding', 'whole', 'down', 'seem', 'whereas', 'to', 
        'their', 'various', 'thereafter', '‘d', 'above', 'put', 'sometime', 'moreover', 'whoever', 'although', 
        'at', 'four', 'each', 'among', 'whatever', 'any', 'anyhow', 'herein', 'become', 'last', 'between', 'still', 
        'was', 'almost', 'twelve', 'used', 'who', 'go', 'not', 'enough', 'well', '’ve', 'might', 'see', 'whose', 
        'everywhere', 'yourselves', 'across', 'myself', 'further', 'did', 'then', 'is', 'except', 'up', 'take', 
        'became', 'however', 'many', 'thence', 'onto', '‘m', 'my', 'own', 'must', 'wherein', 'elsewhere', 'behind', 
        'becomes', 'alone', 'due', 'being', 'neither', 'a', 'over', 'beside', 'fifteen', 'meanwhile', 'upon', 'next', 
        'forty', 'what', 'less', 'and', 'please', 'toward', 'about', 'below', 'hereafter', 'whether', 'yet', 'nor', 
        'against', 'whereupon', 'top', 'first', 'three', 'show', 'per', 'five', 'two', 'ourselves', 'whenever', 
        'get', 'thereby', 'noone', 'had', 'now', 'everyone', 'everything', 'nowhere', 'ca', 'though', 'least', 
        'so', 'both', 'otherwise', 'whereby', 'unless', 'somewhere', 'give', 'formerly', '’d', 'under', 
        'while', 'empty', 'doing', 'besides', 'thus', 'this', 'anyone', 'its', 'after', 'bottom', 'call', 
        'n’t', 'name', 'even', 'eleven', 'by', 'from', 'when', 'or', 'anyway', 'how', 'the', 'all', 
        'much', 'another', 'since', 'hundred', 'serious', '‘ve', 'ever', 'out', 'full', 'themselves', 
        'been', 'in', "'d", 'wherever', 'part', 'someone', 'therein', 'can', 'seemed', 'hereby', 'others', 
        "'s", "'re", 'most', 'one', "n't", 'into', 'some', 'will', 'these', 'twenty', 'here', 'as', 'nobody', 
        'also', 'along', 'than', 'anything', 'he', 'there', 'does', 'we', '’ll', 'latterly', 'are', 'ten', 
        'hers', 'should', 'they', '‘s', 'either', 'am', 'be', 'perhaps', '’re', 'only', 'namely', 'sixty', 
        'made', "'m", 'always', 'those', 'have', 'again', 'her', 'once', 'ours', 'herself', 'else', 'has', 'nine', 
        'more', 'sometimes', 'your', 'yours', 'that', 'around', 'his', 'indeed', 'mostly', 'cannot', '‘ll', 'too', 
        'seems', '’m', 'himself', 'latter', 'whither', 'amount', 'other', 'nevertheless', 'whom', 'for', 'somehow', 
        'beforehand', 'just', 'an', 'beyond', 'amongst', 'none', "'ve", 'say', 'via', 'but', 'often', 're', 'our', 
        'because', 'rather', 'using', 'without', 'throughout', 'on', 'she', 'never', 'eight', 'no', 'hereupon', 
        'them', 'whereafter', 'quite', 'which', 'move', 'thru', 'until', 'afterwards', 'fifty', 'i', 'itself', 'n‘t',
        'him', 'could', 'front', 'within', '‘re', 'back', 'such', 'already', 'several', 'side', 'whence', 'me', 
        'same', 'were', 'it', 'every', 'third', 'together']


def classifying_results(y_test, y_predict, train):
    """
    Input
        y_predict: predicted y values (e.g. array([0, 0, 0, ..., 0, 1, 0], dtype=int64)) 
        y_test: actual y values
            (e.g.
            X = train_tfidf
            y = train['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=123))
        train: an original dataframe (e.g. train = pd.read_csv('../data/train_clean.csv'))

    Output
        true_positive: tweets classified as disaster-related correctly(actual = 1, predict = 1)
        true_negative: tweets classified as nondisaster-related correctly(actual = 0, predict = 0)
        false_negative: tweets misclassified as nondisaster-related(actual = 1, predict = 0)
        false_positive: tweets misclassified as disaster-related(actual = 0, predict = 1)
    """
    correct_list = []
    error_list = []
    for idx, value in enumerate(y_test):
        if y_predict[idx] == value:
            correct_list.append(y_test.index[idx])
        else:
            error_list.append(y_test.index[idx])

    train_correct = train.loc[correct_list]
    train_error = train.loc[error_list]

    true_positive = train_correct[train_correct['target'] == 1]
    true_negative = train_correct[train_correct['target'] == 0]
    false_negative = train_error[train_error['target'] == 1]
    false_positive = train_error[train_error['target'] == 0]

    return true_positive, true_negative, false_negative, false_positive



def analysing_ngrams(false_negative, false_positive, ngrams, N=10):
    """
    Input
        false_negative: tweets misclassified as nondisaster-related (actual = 1, predict = 0)
        false_positive: tweets misclassified as disaster-related(actual = 0, predict = 1)
        ngrams: A contiguous sequence of n words from a given text 
        N: Number of rows you want to show as output

    Output
        A bar chart
    """
    disaster_unigrams = defaultdict(int)
    nondisaster_unigrams = defaultdict(int)

    for tweet in false_negative['text']:
        for word in generate_ngrams(tweet, n_gram=ngrams):
            disaster_unigrams[word] += 1
            
    for tweet in false_positive['text']:
        for word in generate_ngrams(tweet, n_gram=ngrams):
            nondisaster_unigrams[word] += 1
            
    df_disaster_unigrams = pd.DataFrame(sorted(disaster_unigrams.items(), key=lambda x: x[1])[::-1])
    df_nondisaster_unigrams = pd.DataFrame(sorted(nondisaster_unigrams.items(), key=lambda x: x[1])[::-1])

    fig, axes = plt.subplots(ncols=2, figsize=(18, 10), dpi=100)
    plt.tight_layout()

    sns.barplot(y=df_disaster_unigrams[0].values[:N], x=df_disaster_unigrams[1].values[:N], ax=axes[0], color='red')
    sns.barplot(y=df_nondisaster_unigrams[0].values[:N], x=df_nondisaster_unigrams[1].values[:N], ax=axes[1], color='green')

    for i in range(2):
        axes[i].spines['right'].set_visible(False)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=13)
        axes[i].tick_params(axis='y', labelsize=13)
    
    if ngrams == 1:
        G = 'uni'
    elif ngrams == 2:
        G = 'bi'
    elif ngrams == 3:
        G = 'tri'

    axes[0].set_title(f'Top {N} most common {G}grams in False Negative', fontsize=15)
    axes[1].set_title(f'Top {N} most common {G}grams in False Positive', fontsize=15)

    plt.show()


def word_cloud(false_negative, max_words=100):
    """
    Input
        false_negative: You can put any types of classified tweets.
        max_words: Number of words you want to show

    Output
        A twitter bird picture
    """
    corpus_words = create_corpus_df(false_negative)
    corpus_error_disaster = [word for word in corpus_words if not word in STOPWORDS]

    tweet_mask = np.array(Image.open("../data/twitter_mask.png"))

    plt.figure(figsize=(12,8))
    word_cloud = WordCloud(
                            mask=tweet_mask,
                            background_color='white',
                            max_font_size = 100,
                            max_words=max_words,
                            ).generate(" ".join(corpus_error_disaster[:200]))
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show()