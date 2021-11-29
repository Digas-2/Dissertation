# by convention our package is to be imported as dp (for Differential Privacy!)
import pydp as dp
from pydp.algorithms.laplacian import BoundedSum, Count
import pandas as pd
import statistics  # for calculating mean without applying differential privacy
import matplotlib.pyplot as plt
import numpy as np
import math


def cal_private_count_per_day(
    privacy_budget: float, data_list: list
):  # using PyDP library
    private_count = []
    for data in data_list:
        x = Count(privacy_budget)
        count = x.quick_result(data)
        private_count.append(count)
    return private_count


visits_case1 = []  # list for case 1
visits_case2 = []  # list for case 2

#'DRUG_TYPE','DRUG',DOSE_VAL_RX','DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP','ROUTE'

days = ['DRUG']

procedures = pd.read_csv(
    '/home/diogo/Documents/Tese/Datasets/PRESCRIPTIONS.csv')

procedures = procedures.drop(['ROW_ID', 'HADM_ID'], axis=1)


# print(procedures)

procedures = procedures.pivot_table(index='SUBJECT_ID', columns='DRUG',
                                    aggfunc='size', fill_value=0).astype(int)

# print(procedures)


for col in procedures.columns:
    # Case 1: List for max contribution 7 days a week
    visits_case1.append(procedures.loc[procedures[col] > 0][col].tolist())

# print(visits_case1)
# caluculate the private count
# print(visits_case1)
epsilon = math.log(3)
private_count_perDay_week = cal_private_count_per_day(epsilon, visits_case1)

# print(private_count_perDay_week)

x = list(procedures.columns)

plt.scatter(x=x, y=private_count_perDay_week,
            c='orange')

            
plt.savefig('../Analysis/diff_priv_pres_drug.png')
