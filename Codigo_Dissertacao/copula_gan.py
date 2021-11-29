from sdv.tabular import CopulaGAN
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
import numpy as np
import gc
import sys



def pres():
    df3 = pd.read_csv(
        '/home/up201503723/Datasets/physionet.org/files/mimiciii/1.4/PRESCRIPTIONS.csv')
    df3 = df3.drop(['STARTDATE', 'ENDDATE', 'HADM_ID', 'ICUSTAY_ID', 'DRUG_NAME_POE',
                    'DRUG_NAME_GENERIC', 'FORMULARY_DRUG_CD', 'GSN', 'NDC', 'PROD_STRENGTH'], axis=1)

    df3 = df3.dropna()
    df3 = df3.sort_values(by=['SUBJECT_ID'])
    #new_data = pd.DataFrame(columns = list(df3.columns))

    split_df1 = np.array_split(df3, 10)
    for n in range(10):
        df1 = split_df1[n]
        # Funções que criam os dados sintéticos
        model = CopulaGAN()
        model.fit(df1)
        synth_df = model.sample(len(df1.index))
        print(synth_df.head())
        #new_data = new_data.append(synth_df)
        synth_df.to_csv(
            '../Output/mimic_output_prescriptions_copulagan.csv', mode='a', index=False)
        gc.collect()
    # new_data.to_csv('../Output/mimic_output_prescriptions_ctgan.csv',index=False)


def proc():
    df3 = pd.read_csv(
        '/home/up201503723/Datasets/physionet.org/files/mimiciii/1.4/PROCEDURES_ICD.csv')
    df3 = df3.dropna()
    df3 = df3.drop(['HADM_ID', 'SEQ_NUM'], axis=1)

    df3 = df3.sort_values(by=['SUBJECT_ID'])
    #new_data = pd.DataFrame(columns = list(df3.columns))

    # Funções que criam os dados sintéticos
    model = CopulaGAN()
    model.fit(df3)
    synth_df = model.sample(len(df3.index))
    print(synth_df.head())
    #new_data = new_data.append(synth_df)
    synth_df.to_csv('../Output/mimic_output_procedures_copulagan.csv', index=False)
    gc.collect()


def diag():
    df3 = pd.read_csv(
        '/home/up201503723/Datasets/physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv')
    df3 = df3.dropna()
    df3 = df3.drop(['HADM_ID', 'SEQ_NUM'], axis=1)

    df3 = df3.sort_values(by=['SUBJECT_ID'])
    #new_data = pd.DataFrame(columns = list(df3.columns))

    # Funções que criam os dados sintéticos
    split_df1 = np.array_split(df3, 2)
    for n in range(2):
        df1 = split_df1[n]
        # Funções que criam os dados sintéticos
        model = CopulaGAN()
        model.fit(df1)
        synth_df = model.sample(len(df1.index))
        print(synth_df.head())
        #new_data = new_data.append(synth_df)
        synth_df.to_csv(
            '../Output/mimic_output_diagnoses_copulagan.csv', mode='a', index=False)
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
