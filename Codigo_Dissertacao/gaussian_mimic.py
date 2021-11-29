from sdv.tabular import GaussianCopula
import pandas as pd
import numpy as np
import gc
import sys


def pres():

    #df = pd.read_csv('/home/up201503723/Datasets/physionet.org/files/mimiciii/1.4/PROCEDURES_ICD.csv')
    df = pd.read_csv(
        '/home/up201503723/Datasets/physionet.org/files/mimiciii/1.4/PRESCRIPTIONS.csv')
    #df = pd.read_csv('/home/up201503723/Datasets/physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv')
    #df = df.drop(['HADM_ID','SEQ_NUM'], axis=1)
    df = df.drop(['STARTDATE', 'ENDDATE', 'HADM_ID', 'ICUSTAY_ID', 'DRUG_NAME_POE',
                  'DRUG_NAME_GENERIC', 'FORMULARY_DRUG_CD', 'GSN', 'NDC', 'PROD_STRENGTH'], axis=1)
    df = df.dropna()
    df = df.sort_values(by=['SUBJECT_ID'])
    #new_data = pd.DataFrame(columns = list(df.columns))

    split_df1 = np.array_split(df, 10)
    for n in range(10):
        model = GaussianCopula()
        df1 = split_df1[n]
        model.fit(df1)
        synth_df = model.sample(len(df1.index))
        print(synth_df.head())
        #new_data = new_data.append(synth_df)
        synth_df.to_csv(
            '../Output/mimic_output_prescriptions_gaussian.csv', mode='a', index=False)
        gc.collect()


def proc():

    df = pd.read_csv(
        '/home/up201503723/Datasets/physionet.org/files/mimiciii/1.4/PROCEDURES_ICD.csv')
    df = df.dropna()
    df = df.drop(['HADM_ID', 'SEQ_NUM'], axis=1)
    df = df.sort_values(by=['SUBJECT_ID'])
    #new_data = pd.DataFrame(columns = list(df.columns))

    split_df1 = np.array_split(df, 2)
    for n in range(2):
        model = GaussianCopula()
        df1 = split_df1[n]
        model.fit(df1)
        synth_df = model.sample(len(df1.index))
        print(synth_df.head())
        #new_data = new_data.append(synth_df)
        synth_df.to_csv(
            '../Output/mimic_output_procedures_gaussian.csv', mode='a', index=False)
        gc.collect()


def diag():
    pd.options.mode.use_inf_as_na = True
    df = pd.read_csv(
        '/home/up201503723/Datasets/physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv')
    
    df = df.dropna()
    df = df.drop(['HADM_ID', 'SEQ_NUM'], axis=1)
    df = df.sort_values(by=['SUBJECT_ID'])

    split_df1 = np.array_split(df, 10)
    for n in range(10):
        model = GaussianCopula()
        df1 = split_df1[n]
        model.fit(df1)
        synth_df = model.sample(len(df1.index))
        print(synth_df.head())
        #new_data = new_data.append(synth_df)
        synth_df.to_csv(
            '../Output/mimic_output_diagnoses_gaussian.csv', mode='a', index=False)
        gc.collect()


def main():
    print(sys.argv[1])
    if(sys.argv[1] == 'diag'):
        diag()
    elif(sys.argv[1] == 'proc'):
        proc()
    else:
        pres()


if __name__ == "__main__":
    main()
