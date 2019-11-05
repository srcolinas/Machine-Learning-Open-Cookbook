import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_binary_classifcation(X, y=None, model=None, contour_alpha=0.1, **kwargs):
    
    cmap = ListedColormap(['#00aeff', '#ff8400'])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    x2_min, x2_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01)
    )
    
    if model is not None:
        X_ = np.array([xx1.ravel(), xx2.ravel()]).T
        X_ = np.concatenate((np.ones((X_.shape[0], 1)), X_), axis=1)
    
        z = model.predict(X_, **kwargs)
        z = np.reshape(z, xx1.shape)
        plt.contourf(xx1, xx2, z, alpha=contour_alpha, cmap=cmap)
    
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap)