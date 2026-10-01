
import numpy as np
import faiss
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import helpers as hp
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from knn import KNNClassifier
np.random.seed(0)
class KNNClassifier:
    def __init__(self, k, distance_metric='l2'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.Y_train = None

    def fit(self, X_train, Y_train):
        """
        Update the kNN classifier with the provided training data.

        Parameters:
        - X_train (numpy array) of size (N, d): Training feature vectors.
        - Y_train (numpy array) of size (N,): Corresponding class labels.
        """
        self.X_train = X_train.astype(np.float32)
        self.Y_train = Y_train
        d = self.X_train.shape[1]
        if self.distance_metric == 'l2':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L2)
        elif self.distance_metric == 'l1':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L1)
        else:
            raise NotImplementedError
        pass
        self.index.add(self.X_train)

    def predict(self, X):
        """
        Predict the class labels for the given data.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.

        Returns:
        - (numpy array) of size (M,): Predicted class labels.
        """
        #### YOUR CODE GOES HERE ####
        # Calls knn_distance, which handles type conversion and neighbor search.
        D, I = self.knn_distance(X)
        
        predictions = np.zeros(X.shape[0], dtype=self.Y_train.dtype)
        
        # Perform majority vote using the indices I
        for i in range(I.shape[0]):
            labels = self.Y_train[I[i]]
            predictions[i] = np.bincount(labels).argmax()
            
        return predictions

    def knn_distance(self, X):
        """
        Calculate kNN distances for the given data. You must use the faiss library to compute the distances.
        See lecture slides and https://github.com/facebookresearch/faiss/wiki/Getting-started#in-python-2 for more information.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.

        Returns:
        - (numpy array) of size (M, k): kNN distances.
        - (numpy array) of size (M, k): Indices of kNNs.
        """
        X = X.astype(np.float32)
	#### YOUR CODE GOES HERE ####
        D, I = self.index.search(X, self.k)
        if self.distance_metric == 'l2':
            D = np.sqrt(D)
        return D, I


def load_data():
    """Reads data, separates features (long, lat) and labels (state), and returns 
    them as integer NumPy arrays."""
    train_x = pd.read_csv('C:\\Users\\hadar\\IML\\EX2\\train.csv')
    X_train = train_x[['long', 'lat']].values
    Y_train = train_x['state'].values.astype(int)

    test_x = pd.read_csv('C:\\Users\\hadar\\IML\\EX2\\test.csv')
    X_test = test_x[['long', 'lat']].values
    Y_test = test_x['state'].values.astype(int)

    return X_train, Y_train, X_test, Y_test


def evaluate_knn_grid_search():
    """
    Performs a grid search over specified k values and distance metrics (L1, L2)
    to evaluate the KNNClassifier model performance.
    
    Returns:
        - result (np.array): 5x2 array of test accuracies.
        - table (pd.DataFrame): Formatted table of results.
        - models_info (list): List of dictionaries containing model, k, metric, and accuracy.
    """
    
    # --- Data and Hyperparameter Setup ---
    X_train, Y_train, X_test, Y_test = load_data()

    k_list = [1, 10, 100, 1000, 3000]
    metrics = ['l1', 'l2']
    result = np.zeros((len(k_list), len(metrics)))

    # List to store all 10 trained models and their performance
    models_info = []

    # ---Grid Search, Training, and Evaluation ---
    for i, k in enumerate(k_list):
        for j, metric in enumerate(metrics):
            
            # 1. Initialize and Train the Model
            clf = KNNClassifier(k, metric)
            clf.fit(X_train, Y_train)

            # 2. Evaluate Performance
            y_pred = clf.predict(X_test)
            acc = np.mean(y_pred == Y_test)

            # 3. Store Results
            result[i, j] = acc
            models_info.append({
                'k': k,
                'metric': metric,
                'accuracy': acc,
                'model': clf
            })

    # ---Presentation and Return ---
    table = pd.DataFrame(result, index=k_list, columns=metrics)
    print("\nAccuracy Table (rows = k, cols = distance metric):")
    print(table)

    return result, table, models_info

def plot_results():
    X_train, Y_train, X_test, Y_test = load_data()
    clf_k_max_l2 = KNNClassifier(1, 'l2')
    clf_k_max_l2.fit(X_train, Y_train)
    hp.plot_decision_boundaries(clf_k_max_l2, X_test, Y_test)

    clf_k_min_l2 = KNNClassifier(3000, 'l2')
    clf_k_min_l2.fit(X_train, Y_train)
    hp.plot_decision_boundaries(clf_k_min_l2, X_test, Y_test)

    clf_k_max_l1 = KNNClassifier(1, 'l1')
    clf_k_max_l1.fit(X_train, Y_train)
    hp.plot_decision_boundaries(clf_k_max_l1, X_test, Y_test)


def anomaly_detection():
    X_train, _, _, _ = load_data() 
    if X_train is None: return

    try:
        ad_test = pd.read_csv('C:\\Users\\hadar\\IML\\EX2\\AD_test.csv')
    except FileNotFoundError:
        print("Error: AD_test.csv not found at the specified path.")
        return

    ad_coords = ad_test[['long', 'lat']].values
    K_NEIGHBORS = 5
    K_ANOMALIES = 50 

    # Fit the classifier for distance computation (using L2)
    clf = KNNClassifier(K_NEIGHBORS, 'l2') 
    clf.fit(X_train, np.zeros(X_train.shape[0])) 

    # Compute 5-NN distances and sum them for the anomaly score
    distances, _ = clf.knn_distance(ad_coords) 
    sum_of_distances = np.sum(distances, axis=1)

    # Find the indices of the 50 points with the highest scores
    sorted_indices = np.argsort(sum_of_distances)[::-1] 
    top_anomalies_indices = sorted_indices[:K_ANOMALIES]

    # --- Plotting ---
    anomalous_coords = ad_coords[top_anomalies_indices]
    normal_mask = np.ones(ad_coords.shape[0], dtype=bool)
    normal_mask[top_anomalies_indices] = False
    normal_coords = ad_coords[normal_mask]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_train[:, 0], X_train[:, 1], color='black', alpha=0.01, s=5, label='Train data (Black)')
    plt.scatter(anomalous_coords[:, 0], anomalous_coords[:, 1], color='red', s=10, label='Top 50 Anomalies (Red)')
    plt.scatter(normal_coords[:, 0], normal_coords[:, 1], color='blue', s=5, label='Normal Points (Blue)')
    plt.title("Anomaly Detection using kNN (k=5, L2 distance score)")
    plt.legend()
    plt.show()
 
    

def load_data_with_validation():
    """
    Loads data and splits the training set into training (80%) and validation (20%) sets.
    Assumes train.csv and test.csv are in the current working directory.
    """
    try:
        train_x = pd.read_csv('C:\\Users\\hadar\\IML\\EX2\\train.csv')
        test_x = pd.read_csv('C:\\Users\\hadar\\IML\\EX2\\test.csv')
    except FileNotFoundError:
        print("Error: train.csv or test.csv not found.")
        return None, None, None, None, None, None

    # 1. Load initial data
    X_full = train_x[['long', 'lat']].values
    Y_full = train_x['state'].values.astype(int)
    X_test = test_x[['long', 'lat']].values
    Y_test = test_x['state'].values.astype(int)

    # 2. Split X_full into Training (80%) and Validation (20%) sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_full, Y_full, 
        test_size=0.2, 
        random_state=42, # Use a fixed seed for reproducibility
        stratify=Y_full # Ensures class proportions are maintained
    )

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def decision_tree_classification():
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data_with_validation()
    if X_train is None: return
    # Hyperparameters for Task 6.1
    max_depths = [1, 2, 4, 6, 10, 20, 50, 100]
    max_leaf_nodes_list = [50, 100, 1000]

    # List to store the results of all 24 models
    tree_models_info = []

    for max_depth in max_depths:
        for max_leaf_nodes in max_leaf_nodes_list:
            clf = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=42)
            clf.fit(X_train, Y_train)
            acc_train = clf.score(X_train, Y_train)
            acc_val = clf.score(X_val, Y_val)
            acc_test = clf.score(X_test, Y_test)
            tree_models_info.append({
                'max_depth': max_depth,
                'max_leaf_nodes': max_leaf_nodes,
                'acc_train': acc_train,
                'acc_val': acc_val,
                'acc_test': acc_test,
                'model': clf
            })
    print("Grid Search Complete. 24 models and metrics saved.")
    
    results_df = pd.DataFrame(tree_models_info)
    
    # Print the required sample result table
    print("\nDecision Tree Grid Search Results (Task 6.1 - Sample):")
    results_df_sorted = results_df.sort_values(by='acc_val', ascending=False)
    # Print the top 10 models based on validation accuracy
    print(results_df_sorted[['max_depth', 'max_leaf_nodes', 'acc_train', 'acc_val', 'acc_test']].to_markdown(index=False))

    return tree_models_info, results_df


def load_data_full():
    """
    Loads the full original training data (train.csv) without splitting 
    into train/validation/test sets. This is used for final visualization.
    """
    try:
        train_x = pd.read_csv('C:\\Users\\hadar\\IML\\EX2\\train.csv')
    except FileNotFoundError:
        print("Error: train.csv not found for full data load.")
        return None, None

    X_full = train_x[['long', 'lat']].values
    Y_full = train_x['state'].values.astype(int)

    return X_full, Y_full
    
def best_decision_tree():
    tree_models_info, results_df = decision_tree_classification()
    
    if tree_models_info is None:
        return

    # Find the best model based on Validation Accuracy (Task 6.1)
    best_model_info = results_df.loc[results_df['acc_val'].idxmax()]
    best_clf = best_model_info['model']
    
    # Load the full training data for a representative visualization
    X_full, Y_full = load_data_full()
    
    # Generate Plot (Task 6.2 Q4)
    hp.plot_decision_boundaries(
        model=best_clf, 
        X=X_full, 
        y=Y_full, 
        title=f"Decision Tree Boundaries: Depth={best_model_info['max_depth']}, Leaves={best_model_info['max_leaf_nodes']}"
    )
    
    print("\nDecision Tree Analysis Complete.")
    print("Best Model Hyperparameters:")
    print(best_model_info[['max_depth', 'max_leaf_nodes', 'acc_val']])


def restricted_leaves():
    tree_models_info, results_df = decision_tree_classification()
    
    if tree_models_info is None:
        return
    
    restricted_df = results_df[results_df['max_leaf_nodes'] == 50]
    best_model_info = restricted_df.loc[restricted_df['acc_val'].idxmax()]
    best_clf = best_model_info['model']
    X_full, Y_full = load_data_full()
    hp.plot_decision_boundaries(
        model=best_clf, 
        X=X_full, 
        y=Y_full, 
        title=f"Decision Tree Boundaries: Leaves={best_model_info['max_leaf_nodes']}"
    )
    print("\nRestricted Leaves Analysis Complete.")

def restricted_depth():
    tree_models_info, results_df = decision_tree_classification()
    
    if tree_models_info is None:
        return
    
    restricted_df = results_df[results_df['max_depth'] <= 6 ]
    best_model_info = restricted_df.loc[restricted_df['acc_val'].idxmax()]
    best_clf = best_model_info['model']
    X_full, Y_full = load_data_full()
    hp.plot_decision_boundaries(
        model=best_clf, 
        X=X_full, 
        y=Y_full, 
        title=f"Decision Tree Boundaries: Depth={best_model_info['max_depth']}"
    )
    print("\nRestricted Depth Analysis Complete.")

def random_forest():
    X_full, Y_full = load_data_full()
    clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    clf.fit(X_full, Y_full)
    hp.plot_decision_boundaries(
        model=clf, 
        X=X_full, 
        y=Y_full, 
        title=f"Random Forest Boundaries: Depth={6}, Leaves={300}"
    )
    print("\nRandom Forest Analysis Complete.")

def xgboost():
    from xgboost import XGBClassifier
    X_full, Y_full = load_data_full()
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data_with_validation()
    if X_train is None:
        print("Data load failed.")
        return None
    X_train_plus_val = np.vstack((X_train, X_val))
    Y_train_plus_val = np.hstack((Y_train, Y_val))

    xgboost_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=4, random_state=42)
    xgboost_model.fit(X_train_plus_val, Y_train_plus_val)
    acc_test = xgboost_model.score(X_test, Y_test)
    print(f"XGBoost Accuracy: {acc_test*100:.2f}%")

    xgboost_model_best = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=4, random_state=42)
    xgboost_model_best.fit(X_train_plus_val, Y_train_plus_val)

    hp.plot_decision_boundaries(
        model=xgboost_model, 
        X=X_full, 
        y=Y_full, 
        title=f"XGBoost Boundaries: Depth={6}, Leaves={300}"
    )
    print("\nXGBoost Analysis Complete.")

    return acc_test

# def main():
    # X_train, Y_train, X_test, Y_test = load_data("train.csv", "test.csv")
    # if X_train in None:
    #     return 
    # all_results = []

if __name__ == '__main__':
    result, table, models_info = evaluate_knn_grid_search()
    plot_results()

    anomaly_detection()
    tree_models_info, results_df = decision_tree_classification()
    best_decision_tree()
    restricted_leaves()
    restricted_depth()
    xgboost()