import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
import json
import time

DATA_PATH = "ml/matches_features.csv"
MODEL_OUT = "ml/best_model.joblib"
METRICS_OUT = "ml/metrics.json"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["y"])
    y = df["y"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = []

    # 1) Logistic Regression (sa scalingom)
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    grid_lr = {"clf__C": [0.1, 1.0, 10.0]}

    # 2) Random Forest
    pipe_rf = Pipeline([("clf", RandomForestClassifier(class_weight="balanced", random_state=42))])
    grid_rf = {"clf__n_estimators": [200, 400],
               "clf__max_depth": [None, 10, 20]}

    # 3) Gradient Boosting
    pipe_gb = Pipeline([("clf", GradientBoostingClassifier(random_state=42))])
    grid_gb = {"clf__n_estimators": [100, 200],
               "clf__learning_rate": [0.05, 0.1],
               "clf__max_depth": [2, 3]}

    # 4) SVM (sa scalingom)
    pipe_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, class_weight="balanced"))
    ])
    grid_svm = {"clf__C": [0.5, 1.0, 2.0],
                "clf__gamma": ["scale", "auto"]}

    # 5) KNN (sa scalingom)
    pipe_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ])
    grid_knn = {"clf__n_neighbors": [5, 15, 31],
                "clf__weights": ["uniform", "distance"]}

    configs = [
        ("LogReg", pipe_lr, grid_lr),
        ("RandomForest", pipe_rf, grid_rf),
        ("GradientBoosting", pipe_gb, grid_gb),
        ("SVM", pipe_svm, grid_svm),
        ("KNN", pipe_knn, grid_knn),
    ]

    results = []
    best = None

    for name, pipe, grid in configs:
        print(f"\n=== {name} ===")
        start = time.time()

        gs = GridSearchCV(
            pipe, grid, cv=5,
            scoring="f1_macro",
            n_jobs=-1
        )
        gs.fit(X_train, y_train)

        elapsed = time.time() - start
        y_pred = gs.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred).tolist()

        results.append({
            "model": name,
            "best_params": gs.best_params_,
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "train_time_sec": float(elapsed),
            "confusion_matrix": cm
        })

        print("Best params:", gs.best_params_)
        print("Accuracy:", acc)
        print("F1 macro:", f1)
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        if best is None or f1 > best["f1_macro"]:
            best = {"name": name, "estimator": gs.best_estimator_, "f1_macro": f1}

    # Spremi najbolji
    joblib.dump(best["estimator"], MODEL_OUT)
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump({"results": results, "best_model": best["name"]}, f, indent=2)

    print(f"\nSaved model: {MODEL_OUT}")
    print(f"Saved metrics: {METRICS_OUT}")
    print(f"Best model: {best['name']} (F1_macro={best['f1_macro']:.4f})")

if __name__ == "__main__":
    main()
