"""
Main training script: load → preprocess → train → evaluate
Run with: python src/train.py
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from data_loader import load_data
from preprocess import preprocess_data


def main():
    print("=" * 60)
    print("ML PIPELINE: Diabetes Classification")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1] Loading data...")
    X, y = load_data('data/diabetes.csv')
    
    # Step 2: Preprocess
    print("\n[2] Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Step 3: Train baseline model (Logistic Regression)
    print("\n[3] Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_lr = lr_model.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"\n📊 Logistic Regression Accuracy: {acc_lr:.4f}")
    
    # Step 4: Train Random Forest (optional comparison)
    print("\n[4] Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"\n📊 Random Forest Accuracy: {acc_rf:.4f}")
    
    # Step 5: Detailed classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (Logistic Regression)")
    print("=" * 60)
    print(classification_report(y_test, y_pred_lr))
    
    print("=" * 60)
    print("CLASSIFICATION REPORT (Random Forest)")
    print("=" * 60)
    print(classification_report(y_test, y_pred_rf))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Best Model: {'Random Forest' if acc_rf > acc_lr else 'Logistic Regression'}")
    print(f"Best Accuracy: {max(acc_lr, acc_rf):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()