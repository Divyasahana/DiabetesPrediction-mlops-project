from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from data_loader import load_data
from preprocess import preprocess_data


def main():
    # Load data
    X, y = load_data("data/diabetes.csv")

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Train baseline model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
