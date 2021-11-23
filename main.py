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
# H = 1 and L = 0, apply PCA and other dimension
# reduction techniques to the whole string, and
# then cluster based on the components (reasonable
# choices are n_components = 1, 2, or 3)
mps = raw.loc[:, 'MP1':].replace({'L': 0, 'H': 1})
df = pd.concat([df, mps], axis=1)

# Principal Component Analysis
N_COMPONENTS = 2
X = df.loc[:, 'MP1':].values
pca = PCA(n_components=N_COMPONENTS)
fit = pca.fit_transform(X)

for i in range(N_COMPONENTS):
    df[f'pca_comp{i + 1}'] = [c[i] for c in fit]

# Split into Gayatri and Anustubh stanzas
gayatri, anustubh = df[raw['Meter'] == 'G'], df[raw['Meter'] == 'A']


def pivot(df, k_modes=True):
    """Helper function to pivot dataframe and fill in missing values"""

    values = 'pattern' if k_modes else [f'pca_comp{i + 1}' for i in range(N_COMPONENTS)]
    df = df.pivot(['book', 'poem', 'stanza'], 'pada', values)

    for col in df.columns:
        df[col] = df[col].fillna(0)
        df[col] = df[col].astype('category')

    return df


# Clustering
def _add_k_modes_clusters(df, n_clusters, init='Cao'):
    df = pivot(df)
    km = KModes(n_clusters=n_clusters, init=init, n_init=5, verbose=1)
    clusters = km.fit_predict(df)

    # Add the cluster labels to the dataframe
    df['k_modes'] = km.labels_
    df['k_modes'] = df['k_modes'].astype('category')
    return df


def _add_k_means_clusters(df, n_clusters, scaling=True):
    df = pivot(df, False)
    X = df.values
    if scaling:
        X = preprocessing.StandardScaler().fit_transform(X)
    clusters = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)

    df['k_means'] = clusters
    df['k_means'] = df['k_means'].astype('category')
    return df


def _make_heatmap(df, col, n_rows_plotted=50, n_cols_plotted=20):
    """Make heatmap of clusters.

    Args:
        col: column that stores cluster membership."""

    n_clusters = len(set(df[col]))

    # Plot clusters
    df = df.drop(df.columns[:-1], axis=1).astype('int').unstack()
    cmap = sns.color_palette("pastel", n_clusters)
    ax = sns.heatmap(df.iloc[:n_rows_plotted, :n_cols_plotted], cmap=cmap)

    # Making the color bar discrete
    colorbar = ax.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    ticks = [colorbar.vmin + 0.5 * r / n_clusters + r * i / n_clusters for i in range(n_clusters)]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(string.ascii_uppercase[:n_clusters])

    plt.show()


def main(method='k_means', collection=gayatri, n_clusters=6):
    """Run the clustering algorithm
    
    Args:
        method: "k_means" or "k_modes"
        collection: "gayatri" or "anustubh"
    """
    if method == 'k_means':
        collection = _add_k_means_clusters(collection, n_clusters)
    elif method == 'k_modes':
        collection = _add_k_modes_clusters(collection, n_clusters)

    _make_heatmap(collection, method)
    plt.show()


if __name__ == '__main__':
    main()
