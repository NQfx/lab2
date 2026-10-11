"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    df = pd.read_csv(data.csv_path)

    row_count = df.shape[0]
    missing_by_columns = df.isna().sum().to_dict()
    adults_over_30_count = df[df["Age"] > 30].shape[0]

    mean_age_by_pclass = {}
    for i in df["Pclass"].unique().tolist():
        mean_age_by_pclass[i] = float(df[df["Pclass"] == 1]["Age"].mean())

    survived_count = int(df["Survived"].sum())
    survival_rate_by_pclass = {}

    for i in df["Pclass"].unique().tolist():
        survival_rate_by_pclass[i] = float(df[df["Pclass"] == i]["Survived"].sum() / survived_count)

    highest_fares = df.sort_values("Fare", ascending=False).head(5)["Fare"].tolist()

    return TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_columns,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares
    )

# print(analyze_titanic(TitanicInput(csv_path="titanic.csv")))
