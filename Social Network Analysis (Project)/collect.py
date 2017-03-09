# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:43:11 2016

@author: Vinit
"""


import sys
import time
from TwitterAPI import TwitterAPI
import datetime
import pickle
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen


consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
    
def robust_request(twitter, resource, params, max_tries=5):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)
            
            
def getTweets(twitter,search):
    tweets=[]
    total=3
    for i in range (0,total):
        untilparameter=datetime.datetime.now()-datetime.timedelta(days=i)
        untilparameter=untilparameter.strftime('%Y-%m-%d')
        request = robust_request(twitter,'search/tweets', {'q':search,'count':100,'until':untilparameter,'lang':'en'})
        for r in request:
            tweets.append(r)
    writeToFile('tweets.pkl',tweets)
    return tweets
    pass

def collectuserfriends(twitter,tweets):
    tweetswithfriends=[]
    for req in tweets:
                req['user']['friends']=get_friends(twitter,req['user']['screen_name'])
                tweetswithfriends.append(req)
                writeToFile('data.pkl',tweetswithfriends)
    
def writeToFile(filename,tweets):
    output = open(filename, 'wb')
    pickle.dump(tweets, output)
    output.close()
    
def get_friends(twitter, screen_name):
    list=[]
    request=robust_request(twitter,'friends/ids',{'screen_name':screen_name,'count':5000,'cursor':-1}) #getting the friends list based on candidates, max 5000 at times
    for r in request:
        list.append(r)
    return sorted(list)
    pass
    
        
def download_affin():
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')

    afinn = dict()

    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])
    return afinn
    pass

neg_words = []
pos_words = []

def filllexicondata(afinn):
   pos_words=set([key for key, value in afinn.items() if value>=0])
   output = open('pos.txt', 'wb')
   pickle.dump(pos_words, output)
   output.close()
   neg_words=set([key for key, value in afinn.items() if value<0])
   output = open('neg.txt', 'wb')
   pickle.dump(neg_words, output)
   output.close()
   pass

def main():
    afinn=download_affin()
    filllexicondata(afinn)
    twitter = get_twitter()
    tweets=getTweets(twitter,'demonetization -filter:retweets')
    collectuserfriends(twitter,tweets)


if __name__ == '__main__':
    main()