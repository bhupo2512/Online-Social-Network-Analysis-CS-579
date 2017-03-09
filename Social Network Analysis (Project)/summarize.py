# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 19:19:59 2016

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
from collections import Counter, defaultdict


    
def main():
    summary = open('summary.txt','w')
    fname='tweets.pkl'
    file = open(fname, 'rb')
    tweets = pickle.load(file)
    numUser=set([u['user']['screen_name'] for u in tweets])
    summary.write("Number of users collected:%d\n"%len(numUser))
    summary.write("Number of messages collected:%d\n"%len(tweets))
    fname='communities.pkl'
    file = open(fname, 'rb')
    comp = pickle.load(file)
    summary.write("Number of communities discovered:%d\n"%len(comp))
    numberofcommunities=len(comp)
    communitiesCount = dict(Counter([len(c) for c in comp]))
    summary.write('Average number of users per community:'+ str((sum((comm[0]*comm[1]) for comm in communitiesCount.items())/numberofcommunities)))  
    fname='classification.pkl'
    file = open(fname, 'rb')
    classification = pickle.load(file)
    summary.write("\nNumber of instances per class found:\n")
    summary.write("\nNumber of positive instances per class: %d"%classification['positive_instances_count'])
    summary.write("\nNumber of negative instances per class: %d"%classification['negative_instances_count'])
    summary.write("\nOne example from each class:\n")
    summary.write("\nPositive Class Example:\n%s"%classification['positive_doc'][0].encode('utf8').decode('unicode_escape').encode('ascii','ignore').decode("utf-8"))
    summary.write("\n\nNegative Class Example:\n%s"%classification['negative_doc'][0].encode('utf8').decode('unicode_escape').encode('ascii','ignore').decode("utf-8"))
    
    
    summary.close()
    
if __name__ == '__main__':
    main()