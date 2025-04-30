import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class WeightedKNN:
    def __init__(self, n_neighbors=5):
        """
        Initialize the Weighted KNN model.
        
        Parameters:
        - n_neighbors: Number of nearest neighbors to consider.
        """
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.sample_weight = None
        self.X_train = None
        self.y_train = None
        self.classes_ = None  # Store unique class labels

    def fit(self, X_train, y_train, sample_weight=None):
        """
        Train the Weighted KNN model.
        
        Parameters:
        - X_train: Training features.
        - y_train: Training labels.
        - sample_weight: Weights for each training sample (default: None).
        """
        self.X_train = X_train
        self.y_train = y_train
        self.sample_weight = sample_weight if sample_weight is not None else np.ones(len(y_train))
        self.knn.fit(X_train, y_train)  # Train KNN without sample weights
        self.classes_ = np.unique(y_train)  # Store unique classes

    def predict(self, X_test):
        """
        Predict class labels for test samples using weighted KNN.
        
        Parameters:
        - X_test: Test features.
        
        Returns:
        - Predicted labels for X_test.
        """
        neighbors = self.knn.kneighbors(X_test, return_distance=False)
        predictions = []

        for n_indices in neighbors:
            neighbor_weights = self.sample_weight[n_indices]
            neighbor_labels = self.y_train[n_indices]

            # Weighted vote for classification
            unique_labels, weighted_votes = np.unique(neighbor_labels, return_counts=True)
            weighted_votes = [
                sum(neighbor_weights[neighbor_labels == label])
                for label in unique_labels
            ]

            # Predict class with max weighted vote
            predictions.append(unique_labels[np.argmax(weighted_votes)])

        return np.array(predictions)

    def predict_proba(self, X_test):
        """
        Predict class probabilities for test samples using weighted KNN.
        
        Parameters:
        - X_test: Test features.
        
        Returns:
        - Probability distribution over classes for each test sample.
        """
        neighbors = self.knn.kneighbors(X_test, return_distance=False)
        probas = []

        for n_indices in neighbors:
            neighbor_weights = self.sample_weight[n_indices]
            neighbor_labels = self.y_train[n_indices]

            # Compute weighted sum of votes for each class
            class_prob = {cls: 0 for cls in self.classes_}
            for label, weight in zip(neighbor_labels, neighbor_weights):
                class_prob[label] += weight

            # Normalize to obtain probabilities
            total_weight = sum(class_prob.values())
            if total_weight > 0:
                class_prob = {cls: class_prob[cls] / total_weight for cls in class_prob}

            # Append probabilities in order of classes_
            probas.append([class_prob[cls] for cls in self.classes_])

        return np.array(probas)

