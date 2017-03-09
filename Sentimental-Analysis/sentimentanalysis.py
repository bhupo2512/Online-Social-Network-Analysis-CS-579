# coding: utf-8

"""
CS579: Assignment 2
In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.
You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.
The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.
Complete the 14 methods below, indicated by TODO.
As usual, completing one method at a time, and debugging with doctests, should
help.
"""

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


def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.
    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.
    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.
    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], 
          dtype='<U5')
    """
    ###TODO
    if(keep_internal_punct):
           return np.array([re.sub('^\W+', '',re.sub('\W+$', '',x.lower())) for x in doc.split()])
    else:
         return np.array(re.sub('\W+', ' ', doc.lower()).split())
    pass


def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.
    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.
    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    ###TODO
    for token in tokens:
        token='token='+token
        feats[token]+=1
    pass


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.
    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)
    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.
    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
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


neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.
    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.
    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    ###TODO
    feats['pos_words']=0
    feats['neg_words']=0
    for token in tokens:
        if token.lower() in pos_words:
            feats['pos_words']+=1
        elif token.lower() in neg_words:
            feats['neg_words']+=1
    pass


def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.
    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.
    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    ###TODO
    feats=defaultdict(lambda: 0)
    for function in feature_fns:
        function(tokens,feats)
    listfeatures=sorted(feats.items(),key=lambda tup:(tup[0]))
    return listfeatures
    pass


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.
    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),
    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    ###TODO
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
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).
    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.
    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    ###TODO
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
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.
    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.
    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).
    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])
    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.
      This list should be SORTED in descending order of accuracy.
      This function will take a bit longer to run (~20s for me).
    """
    ###TODO
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


def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    ###TODO
    accuracyList=[]
    for dic in results:
        accuracyList.append(dic['accuracy'])
    accuracyList=sorted(accuracyList)
    plt.plot(range(42), accuracyList,'bo-')
    plt.xlabel('setting')
    plt.ylabel('accuracy')
    plt.savefig('accuracies.png')
    #plt.show()
    pass


def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.
    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    ###TODO
    accuraciesTrue=[]  
    accuraciesFalse=[] 
    mean_accuracylist=[]
    for dic in results:
        if(dic['punct']):
            accuraciesTrue.append(dic['accuracy'])
        if(not dic['punct']):
            accuraciesFalse.append(dic['accuracy'])
    tup=tuple((np.mean(accuraciesTrue),'punct=True'))
    mean_accuracylist.append(tup)
    tup=tuple((np.mean(accuraciesFalse),'punct=False'))
    mean_accuracylist.append(tup)
    accuraciesMin2=[]  
    accuraciesMin5=[]
    accuraciesMin10=[]  
    for dic in results:
        if(dic['min_freq']==2):
             accuraciesMin2.append(dic['accuracy'])
        elif(dic['min_freq']==5):
            accuraciesMin5.append(dic['accuracy'])
        elif(dic['min_freq']==10):
            accuraciesMin10.append(dic['accuracy'])
    tup=tuple((np.mean(accuraciesMin2),'min_freq=2'))
    mean_accuracylist.append(tup)
    tup=tuple((np.mean(accuraciesMin5),'min_freq=5'))
    mean_accuracylist.append(tup)
    tup=tuple((np.mean(accuraciesMin10),'min_freq=10'))
    mean_accuracylist.append(tup)
    for features in set([doc['features'] for doc in results]):
        mean=[]
        for r in results:
            if r['features']==features:
                mean.append(r['accuracy'])
        feature='features='+' '.join([func.__name__ for func in list(features)])
        mean_accuracylist.append((np.mean(mean),feature))
    return sorted(mean_accuracylist, key=lambda x: (-x[0]))
    pass


def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)
    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    ###TODO
    tokens_list = [tokenize(doc,best_result['punct']) for doc in docs]
    functionList=[]
    """for funcname in best_result['features']:
        print(funcname.__name__)
        functionList.append(funcname.__name__ )"""
    X,vocab=vectorize(tokens_list,best_result['features'],best_result['min_freq'])
    clf = LogisticRegression()
    clf.fit(X,labels)
    return clf,vocab
    
    pass


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.
    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    ###TODO
    coef = clf.coef_[0]
    if(label==0):
        top_coef_ind = np.argsort(coef)[:n]
    if(label==1):
        top_coef_ind = np.argsort(coef)[::-1][:n]
    top_coef_terms = np.array([k for k,v in sorted(vocab.items(), key=lambda x: x[1])])[top_coef_ind]
    top_coef = coef[top_coef_ind]
    if(label==0):
        neg=[]
        for f in zip(top_coef_terms, top_coef*-1):
            neg.append(f)
        return neg
    if(label==1):
        pos=[]
        for f in zip(top_coef_terms, top_coef):
           pos.append(f) 
        return pos
    pass
        


def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.
    Note: use read_data function defined above to read the
    test data.
    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    ###TODO
    test_docs, test_labels = read_data(os.path.join('data', 'test'))
    tokens_list = [tokenize(d,best_result['punct']) for d in test_docs]
    X_test,vocab=vectorize(tokens_list,best_result['features'],best_result['min_freq'],vocab)
    return test_docs,test_labels,X_test
    pass


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.
    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.
    Returns:
      Nothing; see Log.txt for example printed output.
    """
    ###TODO
    list1=[]
    predictval = clf.predict(X_test)
    predict_probaval = clf.predict_proba(X_test)
    for predVal in range(len(predictval)):
        dict1 = {}
        if predictval[predVal] != test_labels[predVal]:
            if predictval[predVal] == 0:
                dict1['truth'] = test_labels[predVal]
                dict1['predicted']=predictval[predVal]
                dict1['proba'] = predict_probaval[predVal][0]
                dict1['test'] =test_docs[predVal] 
            else:
                dict1['truth'] = test_labels[predVal]
                dict1['predicted']=predictval[predVal]
                dict1['proba'] = predict_probaval[predVal][1]
                dict1['test'] =test_docs[predVal] 
            list1.append(dict1)
    list1=sorted(list1, key=lambda x: (-x['proba']))[:n]
    for l in list1:
        print('truth=%d predicted=%d proba=%.6f'%(l['truth'],l['predicted'],l['proba']))
        print(l['test']+"\n")
    pass


def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))
    
    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()