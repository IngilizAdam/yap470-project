from sklearn.ensemble import RandomForestClassifier

def fit_random_forest(X_train, y_train, n_estimators):
    # Initialize the random forest classifier
    rf = RandomForestClassifier(n_estimators=n_estimators)

    # Fit the random forest model to the training data
    rf.fit(X_train, y_train)

    # Return the trained random forest model
    return rf