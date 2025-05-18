from dataloader import load_titanic_dataset
from feature_selection import apply_svd, apply_rfe
from model import train_and_evaluate
from sklearn.impute import SimpleImputer

try:
    print("[INFO] Loading dataset...")
    df = load_titanic_dataset()
    print("[INFO] Dataset loaded successfully.")

    y = df['Survived']
    X = df.drop('Survived', axis=1)

    print("[INFO] Imputing missing values...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    print("[INFO] Applying TruncatedSVD...")
    X_svd = apply_svd(X_imputed, n_components=5)
    acc_svd = train_and_evaluate(X_svd, y)
    print(f"\n Accuracy after TruncatedSVD: {acc_svd:.2f}")

    print("[INFO] Applying RFE...")
    X_rfe = apply_rfe(X_imputed, y, n_features_to_select=5)
    acc_rfe = train_and_evaluate(X_rfe, y)
    print(f" Accuracy after RFE: {acc_rfe:.2f}")

except Exception as e:
    print(f"[ERROR] {str(e)}")
