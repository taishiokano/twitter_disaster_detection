import configparser
import tweepy
import pandas as pd
import sys

config_ini = configparser.ConfigParser()
config_ini.read('config.ini', encoding='utf-8')

# API authentification
BEARER_TOKEN = config_ini['DEFAULT']['bearer_token']
auth = tweepy.OAuth2BearerHandler(BEARER_TOKEN)
api = tweepy.API(auth, wait_on_rate_limit=True)

# # Another way of API authentification
# API_KEY = config_ini['DEFAULT']['api_key']
# API_SECRET = config_ini['DEFAULT']['api_key']
# ACCESS_TOKEN = config_ini['DEFAULT']['access_token']
# ACCESS_TOKEN_SECRET = config_ini['DEFAULT']['access_token_secret']
# auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
# auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

def get_tweets(search_query):
    # Get tweets by query
    item_number = 10 # num of tweets to get
    tweets = tweepy.Cursor(api.search_tweets, q=search_query, tweet_mode='extended', result_type="mixed", lang='en').items(item_number)

    # Make a table
    tw_data = []
    for tweet in tweets:
        text_wo_lb = tweet.full_text.replace("\n", " ")
        tw_data.append([tweet.id, tweet.created_at, text_wo_lb])
    labels=['tweet_id', 'tweet_created', 'tweet_text']
    df = pd.DataFrame(tw_data,columns=labels)

    # Output as csv file
    file_name='tw_data.csv'
    df.to_csv(file_name,encoding='utf-8-sig',index=False)

if __name__ == "__main__":
    # Reference (Search Tweets): https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query
    search_query = 'earthquake OR (fire alarm) OR accident' # We can specify search conditions by adding queries like "min_faves:200"
    get_tweets(search_query)
