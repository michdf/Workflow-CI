# pyrefly: ignore [missing-import]
import mlflow
# pyrefly: ignore [missing-import]
import mlflow.sklearn
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("iris-classification")

def train_model(n_estimators, min_samples_split, min_samples_leaf):
    # Load data
    train_set = pd.read_csv("data_preprocessing/train.csv")
    val_set = pd.read_csv("data_preprocessing/val.csv")
    test_set = pd.read_csv("data_preprocessing/test.csv")

    X_train = train_set.drop(columns=["Survived"])
    y_train = train_set["Survived"]
    X_val = val_set.drop(columns=["Survived"])
    y_val = val_set["Survived"]
    X_test = test_set.drop(columns=["Survived"])
    y_test = test_set["Survived"]
    
    with mlflow.start_run():
        mlflow.sklearn.autolog()
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_val)

        # Metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='binary')
        recall = recall_score(y_val, y_pred, average='binary')
        f1 = f1_score(y_val, y_pred, average='binary')

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
    
    print("Training selesai dan metrics berhasil dicatat di MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    train_model(args.n_estimators, args.min_samples_split, args.min_samples_leaf)