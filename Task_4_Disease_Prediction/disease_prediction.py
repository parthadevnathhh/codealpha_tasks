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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, 
    confusion_matrix, roc_curve, roc_auc_score
)


def load_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    candidates = [
        os.path.join(script_dir, "heart.csv.csv"),
        os.path.join(script_dir, "heart.csv"),
        "heart.csv.csv",
        "heart.csv",
    ]

    dataset_path = next((path for path in candidates if os.path.exists(path)), None)

    if not dataset_path:
        raise FileNotFoundError("Dataset file could not be located. Please ensure 'heart.csv' is in your project directory.")

    data = pd.read_csv(dataset_path)
    print(f"Loaded dataset successfully from: {dataset_path}")
    print(f"Dataset Shape: {data.shape[0]} samples, {data.shape[1]} features\n")
    return data, script_dir


def preprocess_data(data):
    target_col = "target" if "target" in data.columns else "output"
    
    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    num_features = [col for col in X.columns if X[col].nunique() > 5]
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

    return X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, preprocessor


def train_and_evaluate(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    classifiers = {
        "Logistic Regression": (LogisticRegression(random_state=42, max_iter=1000), True),
        "Support Vector Machine": (SVC(kernel="rbf", probability=True, random_state=42), True),
        "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), False),
        "XGBoost": (XGBClassifier(eval_metric="logloss", random_state=42), False),
    }

    results = {}
    trained_models = {}
    predictions_dict = {}

    print("MODEL TRAINING AND PERFORMANCE EVALUATION :\n")

    for name, (model, scale_required) in classifiers.items():
        X_tr = X_train_scaled if scale_required else X_train
        X_te = X_test_scaled if scale_required else X_test

        model.fit(X_tr, y_train)
        predictions = model.predict(X_te)
        
        
        probabilities = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy
        trained_models[name] = (model, scale_required)
        predictions_dict[name] = (predictions, probabilities)

        print(f"Algorithm: {name}")
        print(f"Accuracy Score: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(classification_report(y_test, predictions, digits=4))
        

    return results, trained_models, predictions_dict


def generate_leaderboard_chart(results, save_dir):
    summary_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
    summary_df["Accuracy (%)"] = (summary_df["Accuracy"] * 100).round(2)
    summary_df = summary_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

    print("\nMODEL LEADERBOARD RANKING :")
    print(summary_df[["Model", "Accuracy (%)"]].to_string(index=False))
    print()

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=summary_df,
        x="Accuracy (%)",
        y="Model",
        hue="Model",
        palette="mako",
        legend=False,
    )

    plt.title("Disease Prediction Model Accuracy Comparison", fontsize=12, pad=12)
    plt.xlabel("Accuracy Percentage (%)", fontsize=10)
    plt.ylabel("Classifier Algorithm", fontsize=10)
    plt.xlim(0, 110)

    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(
                f"{width:.2f}%",
                (width + 1.5, p.get_y() + p.get_height() / 2.0),
                ha="left",
                va="center",
                fontsize=9,
            )

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "model_accuracy_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Performance comparison plot saved at: {plot_path}")

    return summary_df


def generate_confusion_matrices(predictions_dict, y_test, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, (y_pred, _)) in enumerate(predictions_dict.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            ax=axes[idx],
            cbar=False,
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"]
        )
        axes[idx].set_title(f"Confusion Matrix: {name}", fontsize=11)
        axes[idx].set_xlabel("Predicted Label", fontsize=9)
        axes[idx].set_ylabel("True Label", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "confusion_matrices.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Confusion matrices grid plot saved at: {plot_path}")


def generate_roc_curves(predictions_dict, y_test, save_dir):
    plt.figure(figsize=(9, 6))

    for name, (_, y_proba) in predictions_dict.items():
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "roc_auc_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"ROC-AUC curves plot saved at: {plot_path}")


def predict_patient_status(patient_df, best_model, scale_required, preprocessor):
    if scale_required:
        processed_input = preprocessor.transform(patient_df)
    else:
        processed_input = patient_df

    prediction = best_model.predict(processed_input)[0]

    confidence = None
    if hasattr(best_model, "predict_proba"):
        confidence = best_model.predict_proba(processed_input)[0][1]

    status = "Heart Disease Detected" if prediction == 1 else " No Disease Identified"
    confidence_info = f" (Probability Score: {confidence * 100:.1f}%)" if confidence is not None else ""

    print("\nSAMPLE INFERENCE REPORT :")
    print(f"Diagnostic Result: {status}{confidence_info}")
    return prediction


def main():
    data, save_dir = load_dataset()
    X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, preprocessor = preprocess_data(data)

    results, trained_models, predictions_dict = train_and_evaluate(
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    )

    summary_df = generate_leaderboard_chart(results, save_dir)
    generate_confusion_matrices(predictions_dict, y_test, save_dir)
    generate_roc_curves(predictions_dict, y_test, save_dir)

    top_model_name = summary_df.iloc[0]["Model"]
    best_model, scale_required = trained_models[top_model_name]

    print(f"\nSelected primary deployment model: {top_model_name}")

    sample_patient_record = X_test.iloc[[0]]
    predict_patient_status(sample_patient_record, best_model, scale_required, preprocessor)


if __name__ == "__main__":
    main()