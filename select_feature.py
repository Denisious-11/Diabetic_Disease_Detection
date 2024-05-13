from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



def select_features(x, y):

    # configure to select a subset of features
    fs = SelectKBest(score_func=chi2, k="all")

    # learn relationship from training data
    fs.fit(x, y)

    return fs