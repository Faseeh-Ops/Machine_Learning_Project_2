from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def apply_svd(X, n_components=5):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(X)

def apply_rfe(X, y, n_features_to_select=5):
    model = LogisticRegression(max_iter=300)
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    return rfe.fit_transform(X, y)
