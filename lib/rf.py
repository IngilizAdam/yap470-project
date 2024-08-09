from sklearn.ensemble import RandomForestClassifier

def fit_random_forest(X_train, y_train, max_features):
    # Initialize the random forest classifier
    rf = RandomForestClassifier(max_features=max_features)

    # Fit the random forest model to the training data
    rf.fit(X_train, y_train)

    # Return the trained random forest model
    return rf