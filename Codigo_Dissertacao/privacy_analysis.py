import pandas as pd
import sys
import networkx as nx


def jaccard_similarity(g, h):
    i = set(g).intersection(h)
    return round(len(i) / (len(g) + len(h) - len(i)),3)




def main():
    #Load datasets
    #og_df = pd.read_csv('../Datasets/mimic-iii-clinical-database-1.4/'+sys.argv[1].upper()+'.csv')
    #synt_dataset = pd.read_csv('../Output/mimic_output_'+sys.argv[1]+'_'+sys.argv[2]+'.csv')

    #og_df = og_df.groupby(['SUBJECT_ID'])
    #synt_dataset['GROUP_ID'] = synt_dataset.groupby(['SUBJECT_ID']).grouper.group_info[0]
    #print(synt_dataset.sort_values(by=['GROUP_ID']))

    G = nx.read_gml("../grafos/diagnoses_icd_og.gml")
    H = nx.read_gml("../grafos/diagnoses_icd.gml")

    print(jaccard_similarity(G.edges(), H.edges()))

    print(nx.graph_edit_distance(G,H))

if __name__ == "__main__":
    main()