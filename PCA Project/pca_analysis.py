# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PCAHomework:
    def pca_analysis(self, data):
        # Mean-Centering and Scaling Data
        # Scale Function Expects Samples to be Rows
        scaled_data = StandardScaler().fit_transform(data)

        # Computing Covariance Matrix
        cov_matrix = np.cov(scaled_data)

        print("Covariance Matrix: ")
        print(cov_matrix)

        # Performing Eigen-Decomposition
        eigenval, eigenvec = np.linalg.eig(cov_matrix)

        print("Eigenvalues: ")
        print(eigenval)
        print("Eigenvectors: ")
        print(eigenvec)

        # PCA Object
        pca = PCA()
        pca.fit(scaled_data)
        pca_data = pca.transform(scaled_data)

        # Projecting Data onto PC Axes
        project_data = pca.inverse_transform(pca_data)
        plt.scatter(project_data[:, 0], project_data[:, 1], label = 'Projected Data')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Projected Data onto PC Axes')
        plt.show()

        # Calculating Percent Variance
        per_var = np.round(pca.explained_variance_ratio_* 100, decimals = 1)
        labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

        # Scree Plot
        plt.bar(x = range(1, len(per_var) + 1), height = per_var, tick_label = labels)
        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal Component')
        plt.title('Scree Plot')
        plt.show()

        # Scores Plot
        fig, ax = plt.subplots()
        ax.scatter(pca_data[:, 0], pca_data[:, 1], color='blue')
        ax.set_title('Scores Plot')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.show()

        # Loadings Plot
        loadings = pca.components_

        fig, ax = plt.subplots()
        ax.scatter(loadings[:, 0], loadings[:, 1], color='blue')
        ax.set_title('Loadings Plot')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        for i in range(loadings.shape[0]):
            ax.text(loadings[i, 0], loadings[i, 1], 'x' + str(i + 1))
        plt.show()

# my_instance = PCAHomework()
#
# my_instance.pca_analysis('path to data')
