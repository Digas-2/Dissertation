#from graph_tools import graph_metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from sklearn import metrics
from scipy.stats import binom
import glob
from sdv.metrics.tabular import CSTest, KSTest
from sdv.metrics.tabular import BNLikelihood, BNLogLikelihood, GMLogLikelihood
from sdv.metrics.tabular import CategoricalGeneralizedCAP, CategoricalKNN, CategoricalNB,CategoricalRF, CategoricalEnsemble,CategoricalCAP,CategoricalPrivacyMetric
from sdv.evaluation import evaluate
from suda import suda, find_msu
import gc

iteration_model = 1

def dataset_cleaning(df, opt):
    df = df.dropna()
    if(opt == 'pres'):
        df = df.drop(['ROW_ID','STARTDATE', 'ENDDATE', 'HADM_ID', 'ICUSTAY_ID', 'DRUG_NAME_POE',
                      'DRUG_NAME_GENERIC', 'FORMULARY_DRUG_CD', 'GSN', 'NDC', 'PROD_STRENGTH'], axis=1)
    else:
        df = df.drop(['HADM_ID', 'SEQ_NUM','ROW_ID'], axis=1)
    return df


def draw_scatter(df,title,opt,d_opt):
    global iteration_model
    for col in df.columns:
        print(col)
        if((col != 'SUBJECT_ID') and (col != 'ROW_ID') and (col != 'HADM_ID') and (col != 'SEQ_NUM')):
            print(col)
            s = df[col].value_counts(normalize=True)
            plt.scatter(s.index, s,c=['#1f77b4'],s=1)
            plt.xticks([])
            plt.title(title)
            plt.savefig(f'../Analysis/scatter_{opt}_{col}_{d_opt}_{iteration_model}.png')
            iteration_model = iteration_model + 1
            plt.clf()

def suda_score(df,opt,title):
    global iteration_model
    if(opt != "pres"):
        results = suda(df, columns=['ICD9_CODE'])
        #print(results)
        out = results['dis-suda']
        print(out.value_counts())
        if(results['dis-suda'].nunique() > 1):
            if(results['dis-suda'].nunique() <5):
                out = pd.qcut(out, results['dis-suda'].nunique())
            else:
                out = pd.qcut(out, 5)
            ax = out.value_counts(sort=False).plot.bar(rot=0, color="b")
            ax.set_xticklabels([c[1:-1].replace(","," to") for c in out.cat.categories])
            plt.title(title)
            plt.savefig(f'../Analysis/barplot_suda_{opt}_{iteration_model}.png')
            plt.clf()
    else:
        results = suda(df, columns=['ROW_ID','SUBJECT_ID','DRUG_TYPE','DRUG','DOSE_VAL_RX','DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP','ROUTE'])
        out = results['dis-suda']
        #print(results)
        print(out.value_counts())
        if(results['dis-suda'].nunique() > 1):
            out = pd.qcut(out, 5)
            ax = out.value_counts(sort=False).plot.bar(rot=0, color="b")
            ax.set_xticklabels([c[1:-1].replace(","," to") for c in out.cat.categories])
            plt.title(title)
            plt.savefig(f'../Analysis/barplot_suda_{opt}_{iteration_model}.png')
            plt.clf()


def mutual_info(df1, df2,opt1,opt2):
    df1 = df1.applymap(str)
    df2 = df2.applymap(str)
    print("Finding mutual information")
    with open(f'{opt1}_mutual.csv','a') as fd:
        fd.write(opt2)
        fd.write("\n")
        for col in df1:
            fd.write(str(col) + ":  ")
            fd.write(str(metrics.normalized_mutual_info_score(df1[col], df2[col])))
            fd.write("\n")


def unique_values(df1, df2):
    for col in df1:
        print(col)
        print(df1[col].value_counts())
        print(df2[col].value_counts())


def sdvMetrics(df1,df2,opt1,opt2):
    gc.collect()
    print("SDV metrics")
    print(df1)

    with open(f'{opt1}_metric.csv','a') as fd:
        fd.write(opt2)
        fd.write("\n")

        fd.write(str(CSTest.compute(df1, df2)))
        fd.write("\n")

        print("Calculating BNLikelihood")
        fd.write(str(BNLikelihood.compute(df1, df2)))
        fd.write("\n")

        print("Calculating BNLogLikelihood")
        fd.write(str(BNLogLikelihood.compute(df1, df2)))
        fd.write("\n")

def sdvPrivacyMetrics(df1,df2,opt1,opt2):
    gc.collect()
    print("SDV Privacy metrics")
    with open(f'{opt1}_priv.csv','a') as fd:
        fd.write(opt2)
        fd.write("\n")

        df1 = df1.fillna(0)
        df2 = df2.fillna(0)
        df1 = df1.astype(str)
        df2 = df2.astype(str)

        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)

        print("Calculating CGCAP")
        fd.write(str(CategoricalCAP.compute(df1, df2,key_fields=['SUBJECT_ID'],sensitive_fields=['ICD9_CODE'])))
        fd.write("\n")

        print("Calculating KNN")
        fd.write(CategoricalKNN.compute(df1.fillna(0), df2.fillna(0).values.ravel(),key_fields=['SUBJECT_ID'],sensitive_fields=['ICD9_CODE']))
        fd.write("\n")

        print("Calculating RF")
        fd.write(str(CategoricalRF.compute(df1, df2,key_fields=['SUBJECT_ID'],sensitive_fields=['ICD9_CODE'])))
        fd.write("\n")

        print("Calculating Ensamble")
        fd.write(str(CategoricalEnsemble.compute(df1, df2,key_fields=['SUBJECT_ID'],sensitive_fields=['ICD9_CODE'])))
        fd.write("\n")

def sdvPrivacyMetrics2(df1,df2,opt1,opt2):
    gc.collect()
    print("SDV Privacy metrics")
    with open(f'{opt1}_priv.csv','a') as fd:
        fd.write(opt2)
        fd.write("\n")
        df1 = df1.fillna(0)
        df2 = df2.fillna(0)
        df1 = df1.applymap(str)
        df2 = df2.applymap(str)
        fd.write(str(CategoricalCAP.compute(df1, df2,key_fields=['SUBJECT_ID'],sensitive_fields=['DRUG_TYPE','DRUG','DOSE_VAL_RX','DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP','ROUTE'])))
        fd.write("\n")
        fd.write(CategoricalKNN.compute(df1.fillna(0), df2.fillna(0),key_fields=['SUBJECT_ID'],sensitive_fields=['DRUG_TYPE','DRUG','DOSE_VAL_RX','DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP','ROUTE']))
        fd.write("\n")
        print("Calculating RF")
        fd.write(CategoricalRF.compute(df1, df2,key_fields=['SUBJECT_ID'],sensitive_fields=['DRUG_TYPE','DRUG','DOSE_VAL_RX','DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP','ROUTE']))
        fd.write("\n")
        fd.write(str(CategoricalEnsemble.compute(df1.fillna(0), df2.fillna(0),key_fields=['SUBJECT_ID'],sensitive_fields=['DRUG_TYPE','DRUG','DOSE_VAL_RX','DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP','ROUTE'])))
        fd.write("\n")
    
def define_size(kap1,lap2):
    if(len(kap1)< len(lap2)):
        lap2 = lap2.head(len(kap1))
    elif(len(kap1) > len(lap2)):
        kap1 = kap1.head(len(lap2))
    else:
        return kap1,lap2
    return kap1,lap2


def data_quality_analysis():
    #for filepath in glob.iglob(r'/home/up201503723/Tese/Output/mimic_output_diagnoses_*'):
    for filepath in glob.iglob(r'../Output/mimic_output_diagnoses_*'):
        opt = "diag"
        d_opt = 'synth'
        print(filepath)
        synth_dataset = pd.read_csv(filepath)
        original_dataset = pd.read_csv(
            '/home/up201503723/Datasets/physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv')
        #original_dataset = pd.read_csv(
        #    '/home/diogo/Documents/Tese/Datasets/DIAGNOSES_ICD.csv')
        # Remove untrained columns from original dataset
        original_dataset = dataset_cleaning(original_dataset, opt)
        synth_dataset = synth_dataset.dropna()
        original_dataset,synth_dataset = define_size(original_dataset,synth_dataset)
        synth_dataset = synth_dataset.drop(['ROW_ID'], axis=1)
        # Unique value count
        unique_values(original_dataset, synth_dataset)
        # Mutual Information
        mutual_info(original_dataset, synth_dataset,opt,filepath)
        # Do sdv metrics
        sdvMetrics(original_dataset, synth_dataset,opt,filepath)
        sdvPrivacyMetrics(original_dataset, synth_dataset,opt,filepath)
        # Do scatter of synth
        draw_scatter(synth_dataset,filepath,opt,d_opt)
        #Do the suda score
        suda_score(synth_dataset,opt,filepath)


    for filepath in glob.iglob(r'../Output/mimic_output_procedures_*'):
        original_dataset = pd.read_csv(
            '/home/up201503723/Datasets/physionet.org/files/mimiciii/1.4/PROCEDURES_ICD.csv')
        #original_dataset = pd.read_csv(
        #    '/home/diogo/Documents/Tese/Datasets/PROCEDURES_ICD.csv')
        print(filepath)
        opt = "proc"
        d_opt = 'synth'
        synth_dataset = pd.read_csv(filepath)
        # Remove untrained columns from original dataset
        original_dataset = dataset_cleaning(original_dataset, opt)
        synth_dataset = synth_dataset.dropna()
        original_dataset,synth_dataset = define_size(original_dataset,synth_dataset)
        synth_dataset = synth_dataset.drop(['ROW_ID'], axis=1)
        # Unique value count
        unique_values(original_dataset, synth_dataset)
        # Mutual Information
        mutual_info(original_dataset, synth_dataset,opt,filepath)
        # Do sdv metrics
        sdvMetrics(original_dataset, synth_dataset,opt,filepath)
        sdvPrivacyMetrics(original_dataset, synth_dataset,opt,filepath)
        # Do scatter
        draw_scatter(synth_dataset,filepath,opt,d_opt)
        #Do the suda score
        suda_score(synth_dataset,opt)

    for filepath in glob.iglob(r'../Output/mimic_output_prescriptions_*'):
        original_dataset = pd.read_csv(
            '/home/up201503723/Datasets/physionet.org/files/mimiciii/1.4/PRESCRIPTIONS.csv')
        #original_dataset= pd.read_csv(
        #    '/home/diogo/Documents/Tese/Datasets/PRESCRIPTIONS.csv')
        print(filepath)
        opt = "pres"
        synth_dataset = pd.read_csv(filepath)
        d_opt = 'synth'
        # Remove untrained columns from original dataset
        original_dataset = dataset_cleaning(original_dataset, opt)
        synth_dataset = synth_dataset.dropna()
        original_dataset,synth_dataset = define_size(original_dataset,synth_dataset)
        synth_dataset = synth_dataset.drop(['ROW_ID'], axis=1)
        # Unique value count
        unique_values(original_dataset, synth_dataset)
        # Mutual Information
        mutual_info(original_dataset, synth_dataset,opt,filepath)  
        # Do sdv metrics
        sdvMetrics(original_dataset, synth_dataset,opt,filepath)
        sdvPrivacyMetrics2(original_dataset, synth_dataset,opt,filepath)
        # Do scatter
        draw_scatter(synth_dataset,filepath,opt,d_opt)
        #Do the suda score
        suda_score(synth_dataset,opt,filepath)

def sorting_dataset(df):
    return df.sort_values(by=['SUBJECT_ID'])


def check_duplicate(df1, df2):
    print("Finding duplicates")
    df_comparison = df1.merge(df2, indicator=True, how='outer')
    diff_df = df_comparison[df_comparison['_merge'] == 'both']
    return df_comparison, diff_df


def find_equals(df1,df2):
    print("Finding equalss")
    df1 = df1.set_index('SUBJECT_ID', append=True).swaplevel(
        1, 0).sort_index(level=0)
    df2 = df2.set_index('SUBJECT_ID', append=True).swaplevel(
        1, 0).sort_index(level=0)
    
    print(df1.compare(df2, align_axis=0))

def do_og_scatter():
    d_opt = 'og'
    df = pd.read_csv('/home/diogo/Documents/Tese/Datasets/DIAGNOSES_ICD.csv')
    filepath = '/home/diogo/Documents/Tese/Datasets/DIAGNOSES_ICD.csv'
    opt = 'diag'
    df = dataset_cleaning(df,opt)
    draw_scatter(df,filepath,opt,d_opt)

    df = pd.read_csv('/home/diogo/Documents/Tese/Datasets/PROCEDURES_ICD.csv')
    filepath = '/home/diogo/Documents/Tese/Datasets/PROCEDURES_ICD.csv'
    opt = 'proc'
    df = dataset_cleaning(df,opt)
    draw_scatter(df,filepath,opt,d_opt)

    df = pd.read_csv('/home/diogo/Documents/Tese/Datasets/PRESCRIPTIONS.csv')
    filepath = '/home/diogo/Documents/Tese/Datasets/PRESCRIPTIONS.csv'
    opt = 'pres'
    df = dataset_cleaning(df,opt)
    draw_scatter(df,filepath,opt,d_opt)


def main():
    do_og_scatter()
    data_quality_analysis()


if __name__ == "__main__":
    main()