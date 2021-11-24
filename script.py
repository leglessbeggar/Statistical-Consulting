from collections import Counter
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes
import seaborn as sns; sns.set_theme()
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA, PCA, NMF


# Data import
url = 'https://raw.githubusercontent.com/cuboids/consulting/main/dimeter-mp.csv'
raw = pd.read_csv(url)

# Preprocessing
# Split the column `Line` into `book`, `poem`, `stanza` and `pada`
pat = r'(?P<book>\d+)\.(?P<poem>\d+)\.(?P<stanza>\d+)(?P<pada>[a-z])'
df = raw['Line'].str.extract(pat)

# For K-modes:
# Represent the syllables pattern as a binary number.
df['pattern'] = 0
for i in range(1, 9):
    df['pattern'] += (raw[f'MP{i}'] == 'H').values << (8 - i)

# For K-means:
# code H = 1 and L = 0, apply PCA and then
# cluster based on the components (reasonable
# choices are n_components = 1, 2, or 3)
mps = raw.loc[:, 'MP1':].replace({'L': 0, 'H': 1})
df = pd.concat([df, raw['Meter'], mps], axis=1)
# df = pd.concat([df, mps], axis=1)


def pca(df, n_components=2):
    """Principal component analysis"""
    try:
        X = df.loc[:, 'MP1':].values
    except:
        X = df.values
    pca = PCA(n_components=n_components)
    fit = pca.fit_transform(X)

    for i in range(n_components):
        df[f'pca_comp{i + 1}'] = [c[i] for c in fit]

    return df


# Apply pca
df = pca(df)


def pivot(df, mode='k_modes', n_components=2):
    """Helper function to pivot dataframe and fill in missing values"""

    if mode == 'k_means':
        values = [f'pca_comp{i + 1}' for i in range(n_components)]
    elif mode == 'k_modes':
        values = 'pattern'
    else:
        raise ValueError(f'Mode {mode} not supported!')

    df = df.pivot(['book', 'poem', 'stanza', 'Meter'], 'pada', values)
    # df = df.pivot(['book', 'poem', 'stanza'], 'pada', values)

    for col in df.columns:
        df[col] = df[col].fillna(0)
        df[col] = df[col].astype('category')

    return df


# Clustering
def add_k_modes_clusters(df, n_clusters, return_inertia=False):
    df = pivot(df, 'k_modes')
    X = df.values
    print(df)
    if return_inertia:
        return KModes(n_clusters=n_clusters, init='Cao', n_init=5, verbose=1).fit(X).cost_

    km = KModes(n_clusters=n_clusters, init='Cao', n_init=5, verbose=1)
    clusters = km.fit_predict(df)
    df['k_modes'] = km.labels_
    df['k_modes'] = df['k_modes'].astype('category')
    return df


def add_k_means_clusters(df, n_clusters, return_inertia=False):
    df = pivot(df, 'k_means')
    X = df.values
    if return_inertia:
        return KMeans(n_clusters=n_clusters, random_state=0).fit(X).inertia_

    clusters = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)
    df['k_means'] = clusters
    df['k_means'] = df['k_means'].astype('category')
    return df


def make_heatmap(df, col, n_rows_plotted=50, n_cols_plotted=20):
    """Make heatmap of clusters.

    Args:
        col: column that stores cluster membership."""

    n_clusters = len(set(df[col]))

    # Plot clusters
    df = df.drop(df.columns[:-1], axis=1).astype('int').unstack()
    cmap = sns.color_palette("pastel", n_clusters)
    ax = sns.heatmap(df.iloc[:n_rows_plotted, :n_cols_plotted], cmap=cmap)
    ax.set(xticklabels=range(20))
    ax.set(xlabel="stanza")


    # Making the color bar discrete
    colorbar = ax.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    ticks = [colorbar.vmin + 0.5 * r / n_clusters + r * i / n_clusters for i in range(n_clusters)]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(string.ascii_uppercase[:n_clusters])

    plt.show()


def add_elbow_plot(df):
    print(df)
    inertia = [None, None]
    for n_clusters in range(2, 31):
        df_copy = df.copy()
        inertia.append(add_k_means_clusters(df_copy, n_clusters, return_inertia=True))
    plt.plot(inertia)
    plt.xlabel("Number of clusters")
    plt.ylabel("Cost")
    plt.title('Elbow Plot for K-Means')

    plt.show()


def aggregrate_clusters(df):
    df_copy = pd.DataFrame(df).reset_index()
    print(df_copy.group_by('book', axis=1))


def main(df, method='k_means', n_clusters=8):
    """Run the clustering algorithm
    
    Args:
        method: "k_means" or "k_modes"
    """
    if method == 'k_means':
        df = add_k_means_clusters(df, n_clusters)
    elif method == 'k_modes':
        df = add_k_modes_clusters(df, n_clusters)

    make_heatmap(df, method)
    plt.show()


if __name__ == '__main__':
    # df = add_k_means_clusters(df, 8)
    # main(df)
    # plt.show()

    if ...:
        df = add_k_modes_clusters(df, 8)
        df = df.loc[:, 'k_modes'].reset_index()

        x, y, hue = 'k_means', '', 'Meter'
        hue_order = ["A", "G"]

        fig = plt.figure()
        fig.subplots_adjust(hspace=1., wspace=0.4)

        for i, book in enumerate(['2', '3', '7', '5', '10', '8']):
            temp = (df[df['book'] == book]['k_modes']
                    .groupby(df[hue])
                    .value_counts(normalize=True)
                    .rename(y)
                    .reset_index())

            ax = fig.add_subplot(3, 2, i + 1)
            sns.barplot(x='level_1', y=y, hue=hue, data=temp, ax=ax)
            ax.set_xlabel(f'Book {book}')
            plt.xticks = 'abcdefgh'
            if i > 0:
                ax.get_legend().remove()

        plt.show()
