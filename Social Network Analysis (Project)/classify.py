# coding: utf-8


# No imports allowed besides these.
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request
import pickle



def read_data(path):
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    ###TODO
    if(keep_internal_punct):
           return np.array([re.sub('^\W+', '',re.sub('\W+$', '',x.lower())) for x in doc.split()])
    else:
         return np.array(re.sub('\W+', ' ', doc.lower()).split())
    pass


def token_features(tokens, feats):
    for token in tokens:
        token='token='+token
        feats[token]+=1
    pass


def token_pair_features(tokens, feats, k=3):
    ###TODO
    window=[]
    for i in range(0,len(tokens)): 
        j=i
        count=0
        if(i+k-1<len(tokens)):
            while(count<k):
                window.append(tokens[j])
                count+=1
                j+=1
        else:
            break
        getcombination(window,feats)
        window.clear()
    pass


def getcombination(window,feats):
    for i in range(0,len(window)): 
        for j in range(i+1,len(window)):
            token="token_pair="+window[i]+"__"+window[j]
            feats[token]+=1
    pass



def lexicon_features(tokens, feats):
    file = open('pos.txt', 'rb')
    pos_words = pickle.load(file)
    #print(pos_words)
    file.close()
    file = open('neg.txt', 'rb')
    neg_words = pickle.load(file)
    file.close()
    feats['pos_words']=0
    feats['neg_words']=0
    for token in tokens:
        if token.lower() in pos_words:
            feats['pos_words']+=1
        elif token.lower() in neg_words:
            feats['neg_words']+=1
    pass


def featurize(tokens, feature_fns):
    feats=defaultdict(lambda: 0)
    for function in feature_fns:
        function(tokens,feats)
    listfeatures=sorted(feats.items(),key=lambda tup:(tup[0]))
    return listfeatures
    pass


def vectorize(tokens_list, feature_fns, min_freq=2, vocab=None):
    column=[]
    data=[]
    rows=[]
    row=0 
    featureslist=[]
    feats=defaultdict(lambda: 0)   
    for token in tokens_list:
        feats=featurize(token,feature_fns) 
        featureslist.append(dict(feats)) 
    if(vocab==None):
        freq=defaultdict(lambda: 0)
        vocab=defaultdict(lambda: 0)
        tempVocab=defaultdict(lambda: 0)
        vocabList=[]
        for dictionary in featureslist:
            for key,value in dictionary.items():
                if dictionary[key]>0:
                    freq[key]=freq[key]+1
                if (key not in tempVocab) and (freq[key]>=min_freq):
                    vocabList.append(key)
                    tempVocab[key]=0
        vocabList=sorted(vocabList)
        i=0
        for key in vocabList:
                vocab[key]=i
                i+=1
    for dictionary in featureslist:
        for key,value in dictionary.items():
            if key in vocab:
                column.append(vocab[key])
                rows.append(row)
                data.append(value)
        row+=1
    X=csr_matrix((np.array(data,dtype='int64'), (np.array(rows,dtype='int64'),np.array(column,dtype='int64'))), shape=(row, len(vocab)))
    return X,vocab  
    pass


def accuracy_score(truth, predicted):
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    accuracies=[]
    cv=KFold(len(labels),k)
    for train_ind,test_ind in cv:
        clf.fit(X[train_ind],labels[train_ind])
        predict_val = clf.predict(X[test_ind])
        acc_sc = accuracy_score(labels[test_ind], predict_val)
        accuracies.append(acc_sc)
    mean = np.mean(accuracies)
    return mean
    pass


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    result = []
    for punc in punct_vals:
        tokens_list=[]
        for doc in docs:
            tokens_list.append(tokenize(doc,punc))
        for freq in min_freqs:
            for n in range(1, len(feature_fns)+1):
                for feature in combinations(feature_fns,n):
                    featureList=list(feature)
                    X,vocab = vectorize(tokens_list,featureList,freq)
                    dic={}
                    accuracy = cross_validation_accuracy(LogisticRegression(), X, labels, 5)
                    dic['features']=feature
                    dic['punct']=punc
                    dic['accuracy']=accuracy
                    dic['min_freq']=freq
                    result.append(dic)
    return sorted(result,key=lambda k: (-k['accuracy']))         
    pass





def fit_best_classifier(docs, labels, feature_fns):
    tokens_list = [tokenize(doc) for doc in docs]
    X,vocab=vectorize(tokens_list,feature_fns)
    clf = LogisticRegression()
    clf.fit(X,labels)
    return clf,vocab
    pass



def parse_test_data(feature_fns, vocab,tweets):
    tokenslist = [ tokenize(d) for d in tweets ]
    X_test,vocb=vectorize(tokenslist,feature_fns,2,vocab)
    return X_test
    pass

def writeToFile(fname,classification):
        output = open(fname, 'wb')
        pickle.dump(classification, output)
        output.close()

def print_top_predicted(X_test, clf, tweets):
    predicted=clf.predict(X_test)
    predictedoutput=predicted
    outputtweets=tweets[:10]
    for t in zip(predictedoutput,outputtweets):
        if(t[0]==0):
            print("Negative Tweet: "+t[1].encode('utf8').decode('unicode_escape').encode('ascii','ignore').decode("utf-8"))
        elif(t[0]==1):
             print("Positive Tweet: "+t[1].encode('utf8').decode('unicode_escape').encode('ascii','ignore').decode("utf-8"))
        print("\n--------------------------------------------------\n")
    positive_doc=[]
    negative_doc=[]
    positive_instances_count=0
    negative_instances_count=0
    classification={}
    for t in zip(predicted,tweets):
        if t[0]==0:
            negative_doc.append(t[1])
            negative_instances_count+=1
        elif t[0]==1:
            positive_instances_count+=1
            positive_doc.append(t[1])
    classification['positive_instances_count']=positive_instances_count
    classification['negative_instances_count']=negative_instances_count 
    classification['positive_doc']=positive_doc
    classification['negative_doc']=negative_doc   
    writeToFile('classification.pkl',classification)
    pass
    
    


def readFromFile(fname):
    file = open(fname, 'rb')
    tweets = pickle.load(file)
    return tweets

def main():
    feature_fns = [token_pair_features, lexicon_features]
    docs, labels = read_data(os.path.join('data', 'train'))
    fname='tweets.pkl'
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    clf, vocab = fit_best_classifier(docs, labels,feature_fns)
    tweets=readFromFile(fname)
    uniquetweets = set()
    for t in tweets:
        uniquetweets.add(t['text'])
    uniquetweets=list(uniquetweets)
    X_test = parse_test_data(feature_fns, vocab,uniquetweets)
    print('\nPrinting Classified top 10 Tweets based on negative and positive sentiments:\n')
    print_top_predicted(X_test, clf,list(uniquetweets))


if __name__ == '__main__':
    main()