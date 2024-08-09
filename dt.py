from sklearn.tree import DecisionTreeClassifier

def fit_decision_tree(X, y, max_depth):
    # Initialize the decision tree classifier
    clf = DecisionTreeClassifier(max_depth=max_depth)
    
    # Fit the decision tree on the training data
    clf.fit(X, y)
    
    # Return the fitted decision tree classifier
    return clf