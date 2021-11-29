import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd 
import operator
from networkx.algorithms import community
from scipy.stats import wasserstein_distance


def create_graph(df1,graph_dir):
    df1 = df1.drop(['ROW_ID','HADM_ID','SEQ_NUM'], axis=1)
    df1['SUBJECT_ID'] = 'sub_' + df1['SUBJECT_ID'].astype(str)
    df1['ICD9_CODE'] = 'icd9_' + df1['ICD9_CODE'].astype(str)
    print(df1)
    G = nx.from_pandas_edgelist(df1,'SUBJECT_ID','ICD9_CODE')

    for n in list(G):
        if('icd9' in n):
            G.nodes[n]['tag'] = 'icd'
        elif('sub' in n):
            G.nodes[n]['tag'] = 'sub'

    nx.write_gml(G, graph_dir)

def join_graphs(graph_dir_1,graph_dir_2):
    return nx.compose(G,H)


def draw_graph(graph_dir):
    G = nx.read_gml(graph_dir)
    pos = nx.spring_layout(G, iterations=200) 
    nx.draw(G, pos)
    plt.show()

def gephi_output_analysis():
    gephi_df = pd.read_csv('/home/diogo/Documents/Tese/grafos_output/comunity_diagnosis_icd9.csv')
    gephi_df = gephi_df.drop(['timeset','Id'], axis=1)
    #print(gephi_df.sort_values(by=['modularity_class']))
    print(gephi_df.value_counts(subset = 'modularity_class'))


def graph_compare(df1,df2):
    df1 = df1.drop(['ROW_ID','HADM_ID','SEQ_NUM'], axis=1)
    df2 = df2.drop(['ROW_ID','HADM_ID','SEQ_NUM'], axis=1)
    print(df1[df1.isna().any(axis=1)])
    print(df2[df2.isna().any(axis=1)])
    df1 = df1.dropna()
    df2 = df2.dropna()

    df1['ICD9_CODE'] = df1['ICD9_CODE'].str.replace('E','69')
    df1['ICD9_CODE'] = df1['ICD9_CODE'].str.replace('V','86')
    df2['ICD9_CODE'] = df2['ICD9_CODE'].str.replace('V','86')
    df2['ICD9_CODE'] = df2['ICD9_CODE'].str.replace('E','69')

    list1 = df1['SUBJECT_ID'].tolist()
    list2 = df1['ICD9_CODE'].tolist()
    list3 = df2['SUBJECT_ID'].tolist()
    list4 = df2['ICD9_CODE'].tolist()
    print(wasserstein_distance(list1,list3))
    print(wasserstein_distance(list2,list4))

def graph_metrics(graph_dir):


    G = nx.read_gml(graph_dir)
    print(nx.info(G))
    #print('betweenness:' + str(nx.betweenness_centrality(G))) # Run betweenness centrality
    #print('centrality' + str(nx.eigenvector_centrality(G))) # Run eigenvector centrality
    #Communities
    #print("Communities:")
    #print(community.greedy_modularity_communities(G))
    #Density
    #print("Density:")
    #print(nx.density(G))

    #Degree
    degree_dict = dict(G.degree(G.nodes()))
    sorted_degree = sorted(degree_dict.items(), key=operator.itemgetter(1), reverse=True)
    print("Top 20 nodes by degree:")
    for d in sorted_degree[:20]:
        print(d)

