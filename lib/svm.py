from sklearn import svm

def fit_svm(X, y):
    # Initialize the SVM classifier
    clf = svm.SVC()

    # Fit the SVM classifier to the training data
    clf.fit(X, y)

    # Return the trained SVM classifier
    return clf