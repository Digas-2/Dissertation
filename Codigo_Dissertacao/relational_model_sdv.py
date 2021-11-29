from sdv import Metadata, metadata
from sdv.relational import HMA1
import pandas as pd

metadata = Metadata()

df1 = pd.read_csv('/home/diogo/Documents/Tese/Datasets/mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv')
df2 = pd.read_csv('/home/diogo/Documents/Tese/Datasets/mimic-iii-clinical-database-1.4/PROCEDURES_ICD.csv')
df3 = pd.read_csv('/home/diogo/Documents/Tese/Datasets/mimic-iii-clinical-database-1.4/PRESCRIPTIONS.csv')
df1 = df1.drop(['HADM_ID','SEQ_NUM'], axis=1)
df2 = df2.drop(['HADM_ID','SEQ_NUM'], axis=1)
df3 = df3.drop(['HADM_ID','ICUSTAY_ID','DRUG_NAME_POE','DRUG_NAME_GENERIC','FORMULARY_DRUG_CD','GSN','NDC','PROD_STRENGTH'], axis=1)


df3['STARTDATE'] = pd.to_datetime(df3['STARTDATE']).dt.date
df3['ENDDATE'] = pd.to_datetime(df3['ENDDATE']).dt.date
df3['STARTDATE'] = df3['STARTDATE'].astype('datetime64[ns]')
df3['ENDDATE'] = df3['ENDDATE'].astype('datetime64[ns]')
df3['ROUTE'] = df3['ROUTE'].astype('string')
df3['DRUG_TYPE'] = df3['DRUG_TYPE'].astype('string')
df3['DRUG'] = df3['DRUG'].astype('string')
df3['DOSE_UNIT_RX'] = df3['DOSE_UNIT_RX'].astype('string')
df3['FORM_UNIT_DISP'] = df3['FORM_UNIT_DISP'].astype('string')
df3['DOSE_VAL_RX'] = df3['DOSE_VAL_RX'].astype('string')
df3['FORM_VAL_DISP'] = df3['FORM_VAL_DISP'].astype('string')

print(df3.dtypes)

df1 = df1.dropna()
df2 = df2.dropna()
df3 = df3.dropna()

df1 = df1.head(10000)
df2 = df2.head(10000)
df3 = df3.head(10000)

transactions_fields = {
  'STARTDATE': {
      'type': 'datetime',
      'format': '%Y-%m-%d'
  },
  'ENDDATE': {
      'type': 'datetime',
      'format': '%Y-%m-%d'
  }
}

tables =	{
  "diagnoses_icd": df1,
  "procedures_icd": df2,
  "prescriptions": df3
}

metadata.add_table(name='diagnoses_icd',
                    data=tables['diagnoses_icd'],
                    primary_key='SUBJECT_ID')

metadata.add_table(name='procedures_icd',
                    data=tables['procedures_icd'],
                    parent='diagnoses_icd',
                    foreign_key='SUBJECT_ID')

metadata.add_table(name='prescriptions',
                    data=tables['prescriptions'],
                    fields_metadata=transactions_fields,
                    parent='diagnoses_icd',
                    foreign_key='SUBJECT_ID')

print(metadata)
model = HMA1(metadata)
model.fit(tables)

model.save('../Output/relational_model.pkl')

new_data1 = model.sample('diagnoses_icd', sample_children=False)
new_data2 = model.sample('procedures_icd', sample_children=False)
new_data3 = model.sample('prescriptions', sample_children=False)

new_data1.to_csv('../Output/mimic_output_diagnoses_relational_data.csv',index=False)
new_data2.to_csv('../Output/mimic_output_procedures_relational_data.csv',index=False)
new_data3.to_csv('../Output/mimic_output_prescriptions_relational_data.csv',index=False)