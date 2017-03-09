# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:44:07 2016

@author: Vinit
"""


import networkx as nx
import pickle
from itertools import combinations
from networkx import edge_betweenness_centrality as betweenness

            
def readFromFile(fname):
    file = open(fname, 'rb')
    tweets = pickle.load(file)
    file.close()
    return tweets
    
def computeJaccardSimilarity(x,y):
    if len(x)>0 and len(y)>0:
        intersection=set.intersection(*[set(x), set(y)])
        union=set.union(*[set(x), set(y)])
        return len(intersection)/len(union)
    else:
        return 0
    
def graph_creation_for_cluster(tweets):
    graph=nx.Graph()
    users=[usr['user']['screen_name'] for usr in tweets]
    combination=combinations(users, 2)
    for usr in users:
        graph.add_node(usr)
    for comb in combination:
        x=[]
        y=[]
        for usr in tweets:
            if usr['user']['screen_name']==comb[0]:
                x.append(usr['user']['friends'])
        for usr in tweets:
            if usr['user']['screen_name']==comb[1]:
                y.append(usr['user']['friends'])
        coef = computeJaccardSimilarity(x[0],y[0])
        if coef > 0.005:
            graph.add_edge(comb[0],comb[1],weight=coef)
    removenodes = []
    for node,degree in graph.degree().items():
       if degree <= 1:
          removenodes.append(node)          
    graph.remove_nodes_from(removenodes)
    return graph
    

def max_edge(graph):
     centrality = betweenness(graph, weight='weight')
     return max(centrality, key=centrality.get)

def computeCommunities(graph):
    components = [c for c in nx.connected_component_subgraphs(graph)]
    while len(components) < 8:
        edge_to_remove = max_edge(graph)
        graph.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_component_subgraphs(graph)]
    return components
    
def writeToFile(fname,components):
    output = open(fname, 'wb')
    pickle.dump(components, output)
    output.close()

def main():
    fname='data.pkl'
    tweets=readFromFile(fname)
    graph=graph_creation_for_cluster(tweets)
    components=computeCommunities(graph)
    fname='communities.pkl'
    writeToFile(fname,components)
    

if __name__ == '__main__':
    main()