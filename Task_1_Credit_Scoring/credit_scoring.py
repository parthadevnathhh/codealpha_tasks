import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, 
    confusion_matrix, roc_curve, roc_auc_score
)


def load_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    candidates = [
        os.path.join(script_dir, "german_credit_data.csv"),
        os.path.join(script_dir, "german_credit_data.csv.csv"),
        "german_credit_data.csv",
        "german_credit_data.csv.csv",
    ]

    dataset_path = next((path for path in candidates if os.path.exists(path)), None)

    if not dataset_path:
        raise FileNotFoundError(
            "Dataset file could not be located. "
            "Please ensure 'german_credit_data.csv' is inside your Task_1_Credit_Scoring directory."
        )

    data = pd.read_csv(dataset_path)
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])

    print(f"Loaded dataset successfully from: {dataset_path}")
    print(f"Dataset Shape: {data.shape[0]} samples, {data.shape[1]} features\n")
    return data, script_dir


def preprocess_data(data):
    
    if "Risk" in data.columns:
        y = data["Risk"].map({"good": 1, "bad": 0})
        X = data.drop(columns=["Risk"])
    elif "credit_risk" in data.columns:
        y = data["credit_risk"]
        X = data.drop(columns=["credit_risk"])
    else:
        
        target_col = data.columns[-1]
        y = data[target_col]
        X = data.drop(columns=[target_col])

    
    if "Credit amount" in X.columns and "Duration" in X.columns:
        X["Credit_to_Duration_Ratio"] = X["Credit amount"] / (X["Duration"] + 1e-5)

    if "Age" in X.columns:
        X["Age_Group"] = pd.cut(
            X["Age"], 
            bins=[0, 25, 40, 60, 100], 
            labels=["Young", "Adult", "Middle", "Senior"]
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    num_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    cat_features = [col for col in X.columns if col not in num_features]

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, preprocessor


def train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test):
    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    }

    results = {}
    trained_models = {}
    predictions_dict = {}

    print("MODEL TRAINING AND PERFORMANCE EVALUATION \n")

    for name, model in classifiers.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        
        probabilities = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy
        trained_models[name] = model
        predictions_dict[name] = (predictions, probabilities)

        print(f"Algorithm: {name}")
        print(f"Accuracy Score: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(classification_report(y_test, predictions, digits=4))
        

    return results, trained_models, predictions_dict


def generate_visualizations(predictions_dict, y_test, save_dir):
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, (name, (y_pred, _)) in enumerate(predictions_dict.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Greens", ax=axes[idx], cbar=False,
            xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"]
        )
        axes[idx].set_title(f"Confusion Matrix: {name}", fontsize=10)
        axes[idx].set_xlabel("Predicted Risk")
        axes[idx].set_ylabel("True Risk")

    plt.tight_layout()
    cm_path = os.path.join(save_dir, "credit_scoring_confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrices at: {cm_path}")

    
    plt.figure(figsize=(8, 5))
    for name, (_, y_proba) in predictions_dict.items():
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Credit Scoring ROC-AUC Curves')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    roc_path = os.path.join(save_dir, "credit_scoring_roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves at: {roc_path}")


def main():
    data, save_dir = load_dataset()
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, preprocessor = preprocess_data(data)

    results, trained_models, predictions_dict = train_and_evaluate(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    generate_visualizations(predictions_dict, y_test, save_dir)


if __name__ == "__main__":
    main()