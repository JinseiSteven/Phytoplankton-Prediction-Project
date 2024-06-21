import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F


def sum_and_divide_non_nan(row, normalize: bool = True):
    # drop NaN values
    non_nan_values = row.dropna()  
    if normalize:
        return non_nan_values.sum() / len(non_nan_values) if len(non_nan_values) > 0 else np.nan
    else:
        return non_nan_values.sum() if len(non_nan_values) > 0 else np.nan


def aggregate_biotic_data(df: pd.DataFrame, clusters: list[list], fill_method: str ='forward', normalize: bool = True) -> pd.DataFrame:
    """
    Aggregates a biotic data dataframe on time to a few clusters. 
    Input:
        df: A biotic dataframe containing phytoplankton data on time.
        clusters: A list of lists of species.
    Output:
        A usable dataframe with no missing values with the clusters as the columns.
    """
    # check if all the dataframe columns are the same as species in the cluster. 
    if df.columns != [spec for cluster in clusters for spec in cluster]:
        raise ValueError("Not the same species in dataframe as in cluster list.")
    
    NUM_CLUSTERS = len(clusters)
    cluster_dataframes = list()
    output_df = pd.DataFrame(index=df.index, columns=[i for i in range(NUM_CLUSTERS)])
    
    # transform species names clusters to a dataframe
    for cluster in clusters:
        cluster_dataframes.append(df[cluster])
    
    # apply aggregation to a cluster and add to final dataframe
    for i in range(NUM_CLUSTERS):
        cluster_df = cluster_dataframes[i]
        cluster_df['Aggregated'] = cluster_df.apply(sum_and_divide_non_nan(normalize=normalize), axis=1)`

        # apply a fill to nan values based on given fill method
        if fill_method == 'forward':
            cluster_df['Aggregated'] = cluster_df['Aggregated'].ffill()
        elif fill_method == 'backward':
            cluster_df['Aggregated'] = cluster_df['Aggregated'].bfill()
        elif fill_method == 'linear':
            cluster_df['Aggregated'] = cluster_df['Aggregated'].interpolate('linear')
        elif fill_method == 'nearest':
            cluster_df['Aggregated'] = cluster_df['Aggregated'].interpolate('nearest')
        else:
            raise ValueError("Wrong fill_method.")
        
        output_df[i] = cluster_df['Aggregated']

    return output_df


if __name__ == "__main__":
    ...