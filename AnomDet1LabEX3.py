import numpy as np
from scipy.stats import zscore
from sklearn.metrics import balanced_accuracy_score
from pyod.utils.data import generate_data

# 1D dataset, 1000 training points, 10% contamination
X_train, _, y_train, _ = generate_data(n_train=1000, n_test=0, n_features=1, contamination=0.1, random_state=42)

# Z-Score for unidimensional dataset
z_scores = zscore(X_train)

# Z-Score for 10% contamination rate calculation using np.quantile
contamination_rate = 0.1
threshold = np.quantile(np.abs(z_scores), 1 - contamination_rate)

# Classifing data points as anomalies based on Z-Score threshold
y_train_pred = (np.abs(z_scores) > threshold).astype(int)

# Computation of balanced accuracy score
balanced_acc = balanced_accuracy_score(y_train, y_train_pred)
print(f"Z-score threshold: {threshold}")
print(f"Balanced Accuracy: {balanced_acc:.2f}")
