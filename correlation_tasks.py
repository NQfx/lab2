"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations
import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    data_df = pd.read_csv('brainsize.txt', sep="\t", na_values=['NA'])

    cols = data_df.columns.tolist()

    params = cols[1:-1]
    cor_df = pd.DataFrame(index=params, columns=['Male', 'Female'])

    male_data = data_df[data_df['Gender'] == 'Male']
    female_data = data_df[data_df['Gender'] == 'Female']

    for p in params:
        cor_df.loc[p, 'Male'] = male_data[p].corr(male_data['MRI_Count'])
        cor_df.loc[p, 'Female'] = female_data[p].corr(female_data['MRI_Count'])

    return BrainCorrelationSummary(
        men_count=len(male_data),
        women_count=len(female_data),
        women_mri_correlation=cor_df['Female'].to_dict(),
        men_mri_correlation=cor_df['Male'].to_dict(),
        strongest_mri_feature=cor_df.abs().stack().idxmax()[0]
    )

# print(analyze_brain_correlations(BrainDataInput(csv_path="brainsize.txt")))

