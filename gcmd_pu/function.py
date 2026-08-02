from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd


def get_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    p = precisions[:-1]
    r = recalls[:-1]

    f1_scores = 2 * (p * r) / (p + r + 1e-7) # 1e-7 to prevent Nan

    best_index = np.argmax(f1_scores)
    best_thres = thresholds[best_index]
    return best_thres, f1_scores[best_index], p[best_index], r[best_index]

def get_data_by_year(file_path, year_list):
    df = pd.read_parquet(file_path, filters=[('year', 'in', year_list)]).set_index('datetime')
    return df


def optiacal_add_columns(df):
    switch_flag = df['port'].ne(df['port'].shift(1))|df['sample'].ne(df['sample'].shift(1))


    switch_group = switch_flag.cumsum()

    group_start_time = pd.Series(df.index, index=df.index).groupby(switch_group).transform('first')

    df['time_since_switch'] = (df.index -group_start_time ).dt.total_seconds()

    group_start_cycle = df['cycle_time'].groupby(switch_group).transform('first')

    df['cycle_time_diff'] = df['cycle_time'] - group_start_cycle

    return df
