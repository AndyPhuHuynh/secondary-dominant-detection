import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.features.labels import get_label_num_to_string


def plot_pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(
        X_pca[y == 0, 0],
        X_pca[y == 0, 1],
        alpha=0.6, label=f"{get_label_num_to_string(0)}"
    )

    plt.scatter(
        X_pca[y == 1, 0],
        X_pca[y == 1, 1],
        alpha=0.6, label=f"{get_label_num_to_string(1)}"
    )

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.title('PCA of Song Features')
    plt.show()


def plot_tsne(X, y):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        X_tsne[y == 0, 0],
        X_tsne[y == 0, 1],
        alpha=0.6, label=f"{get_label_num_to_string(0)}"
    )

    plt.scatter(
        X_tsne[y == 1, 0],
        X_tsne[y == 1, 1],
        alpha=0.6, label=f"{get_label_num_to_string(1)}"
    )

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.title('t-SNE of Song Features')
    plt.show()


def plot_umap(X, y):
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        X_umap[y == 0, 0],
        X_umap[y == 0, 1],
        alpha=0.6, label=f"{get_label_num_to_string(0)}"
    )

    plt.scatter(
        X_umap[y == 1, 0],
        X_umap[y == 1, 1],
        alpha=0.6, label=f"{get_label_num_to_string(1)}"
    )

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.title('UMAP of Song Features')
    plt.show()



