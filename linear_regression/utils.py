import numpy as np
import matplotlib.pyplot as plt

def make_plot(X_train, y_train, X_test=None, y_test=None, model=None, feature_extractor=None):
    
    _, ax = plt.subplots()
    
    xmin = np.min(X_train)
    xmax = np.max(X_train)

    if X_test is not None:
        xmin = min(np.min(X_test), xmin)
        xmax = max(np.max(X_test), xmax)
        ax.scatter(X_test[:, 0], y_test, label='Test', c='tab:green', alpha=0.6)
    
    if model is not None:
        X_real = np.linspace(xmin, xmax, 1000000).reshape(-1,1)
        if feature_extractor is not None:
            X_real = feature_extractor(X_real)
        y_pred = model.predict(X_real)
        ax.plot(X_real[:, 0], y_pred, label='Model', c='tab:orange')
    
    ax.scatter(X_train[:, 0], y_train, label='Train', c='tab:blue', alpha=0.6)
    
    ymin = y_train.min()
    ymax = y_train.max()
    if y_test is not None:
        ymin = min(ymin, y_test.min())
        ymax = max(ymax, y_test.max())
    ax.set_ylim(ymin, ymax)
    ax.legend()
    plt.show()