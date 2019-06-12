import tweepy as tw
import time

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tw.API(auth)

search_words = ["globalwarminghoax", "globalwarmingisahoax", "climatechange", "climatehustle", "climatechangefraud"]

for search_word in search_words:    
    fp = open(search_word+'.txt', 'a+', encoding='utf8')
    search_query = search_word + " -filter:retweets"
    try:
        pages = tw.Cursor(api.search,
                    q=search_query,
                    lang="en",
                    tweet_mode='extended',
                    count = 100).pages(100)
        for tweets in pages:
            for tweet in tweets:
                fp.write('"' + tweet.full_text.replace('\n', ' ') + '"\n')
    except (tw.TweepError, tw.RateLimitError) as e:
        if e == "[{u'message': u'Rate limit exceeded', u'code': 88}]":
            time.sleep(60*5) #Sleep for 5 minutes
        else:
            print(e)

    fp.close() 
            