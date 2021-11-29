from synthpop import Synthpop
import pandas as pd
import numpy as np
import gc
import sys


def pres():

    #df = pandas.read_csv('../Datasets/mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv')
    df = pd.read_csv(
        '/home/up201503723/Datasets/physionet.org/files/mimiciii/1.4/PRESCRIPTIONS.csv')


    # Synthpop
    # Neste package, é necessário passar à função fit um dos tipos int ou category,
    # Esta informação é passada no dicionário *types*
    #df["ICD9_CODE"] = df["ICD9_CODE"].astype('category')
    df = df.drop(['STARTDATE', 'ENDDATE', 'HADM_ID', 'ICUSTAY_ID', 'DRUG_NAME_POE',
                'DRUG_NAME_GENERIC', 'FORMULARY_DRUG_CD', 'GSN', 'NDC', 'PROD_STRENGTH'], axis=1)
    df = df.sort_values(by=['SUBJECT_ID'])
    df = df.dropna()
    #new_data = pd.DataFrame(columns = list(df.columns))

    df["ROW_ID"] = df["ROW_ID"].astype('int')
    df["SUBJECT_ID"] = df["SUBJECT_ID"].astype('int')
    df["DRUG_TYPE"] = df["DRUG_TYPE"].astype('category')
    df["DRUG"] = df["DRUG"].astype('category')
    df["DOSE_UNIT_RX"] = df["DOSE_UNIT_RX"].astype('category')
    df["FORM_UNIT_DISP"] = df["FORM_UNIT_DISP"].astype('category')
    df["ROUTE"] = df["ROUTE"].astype('category')
    df["FORM_VAL_DISP"] = df["FORM_VAL_DISP"].astype('category')
    df["DOSE_VAL_RX"] = df["DOSE_VAL_RX"].astype('category')

    types = {
        "ROW_ID": "int",
        "SUBJECT_ID": "int",
        "DRUG_TYPE": "category",
        "DRUG": "category",
        "DOSE_VAL_RX": "category",
        "DOSE_UNIT_RX": "category",
        "FORM_VAL_DISP": "category",
        "FORM_UNIT_DISP": "category",
        "ROUTE": "category"
    }


    split_df1 = np.array_split(df, 11)
    for n in range(11):
        spop = Synthpop()
        df1 = split_df1[n]
        print(df1.head())
        # Funções que criam os dados sintéticos
        spop.fit(df1, types)
        synth_df = spop.generate(len(df1))
        print(synth_df.head())
        synth_df.to_csv(
            '../Output/mimic_output_prescriptions_synthpop.csv', mode='a', index=False)
        gc.collect()
    # Output


def proc():

    #df = pandas.read_csv('../Datasets/mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv')
    df = pd.read_csv(
        '/home/up201503723/Datasets/physionet.org/files/mimiciii/1.4/PROCEDURES_ICD.csv')

    # Synthpop
    # Neste package, é necessário passar à função fit um dos tipos int ou category,
    # Esta informação é passada no dicionário *types*
    #df["ICD9_CODE"] = df["ICD9_CODE"].astype('category')
    df = df.dropna()
    df = df.drop(['HADM_ID', 'SEQ_NUM'], axis=1)
    

    df = df.sort_values(by=['SUBJECT_ID'])
    #new_data = pd.DataFrame(columns = list(df.columns))

    types = {
        "ROW_ID": "int",
        "SUBJECT_ID": "int",
        "ICD9_CODE":"category"
    }

    spop = Synthpop()
    spop.fit(df, types)
    synth_df = spop.generate(len(df))
    synth_df.to_csv(
        '../Output/mimic_output_procedures_synthpop.csv', mode='a', index=False)
    gc.collect()
    # Output

def diag():

    #df = pandas.read_csv('../Datasets/mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv')
    df = pd.read_csv(
        '/home/up201503723/Datasets/physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv')

    # Synthpop
    # Neste package, é necessário passar à função fit um dos tipos int ou category,
    # Esta informação é passada no dicionário *types*
    #df["ICD9_CODE"] = df["ICD9_CODE"].astype('category')
    df = df.dropna()
    df = df.drop(['HADM_ID', 'SEQ_NUM'], axis=1)
    

    df = df.sort_values(by=['SUBJECT_ID'])
    #new_data = pd.DataFrame(columns = list(df.columns))

    types = {
        "ROW_ID": "int",
        "SUBJECT_ID": "int",
        "ICD9_CODE":"category"
    }

    spop = Synthpop()
    spop.fit(df, types)
    synth_df = spop.generate(len(df))
    synth_df.to_csv(
        '../Output/mimic_output_procedures_synthpop.csv')
    gc.collect()
    # Output

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
