"""
modelling.py (MLProject version)
Digunakan oleh MLflow Project untuk CI workflow.
"""

import os
import json
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, average_precision_score
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

TRAIN_PATH = 'heart_disease_preprocessing/train.csv'
TEST_PATH  = 'heart_disease_preprocessing/test.csv'
EXPERIMENT = 'Heart-Disease-CI'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators',     type=int,   default=200)
    parser.add_argument('--max_depth',        type=int,   default=5)
    parser.add_argument('--learning_rate',    type=float, default=0.1)
    parser.add_argument('--subsample',        type=float, default=0.8)
    parser.add_argument('--colsample_bytree', type=float, default=0.8)
    parser.add_argument('--random_state',     type=int,   default=42)
    return parser.parse_args()


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    return train.drop('target', axis=1), test.drop('target', axis=1), \
           train['target'], test['target']


def plot_confusion_matrix(y_test, y_pred, save_path):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'], ax=ax)
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_test, y_prob, auc_score, save_path):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#3498db', lw=2, label=f'AUC = {auc_score:.4f}')
    ax.plot([0,1],[0,1], color='gray', lw=1, linestyle='--')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#3498db')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    # ── PENTING: hapus env var yang menyebabkan konflik run ID ──
    for key in ['MLFLOW_RUN_ID', 'MLFLOW_EXPERIMENT_ID']:
        os.environ.pop(key, None)

    print('=' * 55)
    print('  MLFLOW PROJECT - CI Training')
    print('=' * 55)
    print(f'  n_estimators    : {args.n_estimators}')
    print(f'  max_depth       : {args.max_depth}')
    print(f'  learning_rate   : {args.learning_rate}')
    print(f'  subsample       : {args.subsample}')
    print(f'  colsample_bytree: {args.colsample_bytree}')
    print('=' * 55)

    X_train, X_test, y_train, y_test = load_data()
    os.makedirs('tmp_artifacts', exist_ok=True)

    mlflow.set_experiment(EXPERIMENT)

    # Gunakan mlflow.start_run() tanpa run_id agar tidak konflik
    with mlflow.start_run(run_name=f'xgboost_ci_run') as run:
        model = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            random_state=args.random_state,
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        roc_auc   = roc_auc_score(y_test, y_prob)
        avg_prec  = average_precision_score(y_test, y_prob)
        cm        = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        train_acc   = accuracy_score(y_train, model.predict(X_train))

        # Log params
        mlflow.log_param('n_estimators',     args.n_estimators)
        mlflow.log_param('max_depth',        args.max_depth)
        mlflow.log_param('learning_rate',    args.learning_rate)
        mlflow.log_param('subsample',        args.subsample)
        mlflow.log_param('colsample_bytree', args.colsample_bytree)
        mlflow.log_param('random_state',     args.random_state)

        # Log metrics
        mlflow.log_metric('accuracy',      acc)
        mlflow.log_metric('precision',     precision)
        mlflow.log_metric('recall',        recall)
        mlflow.log_metric('f1_score',      f1)
        mlflow.log_metric('roc_auc',       roc_auc)
        mlflow.log_metric('avg_precision', avg_prec)
        mlflow.log_metric('specificity',   specificity)
        mlflow.log_metric('tp',            int(tp))
        mlflow.log_metric('tn',            int(tn))
        mlflow.log_metric('fp',            int(fp))
        mlflow.log_metric('fn',            int(fn))
        mlflow.log_metric('train_accuracy',train_acc)
        mlflow.log_metric('overfit_gap',   train_acc - acc)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path='model')

        # Plots
        cm_path  = 'tmp_artifacts/confusion_matrix.png'
        roc_path = 'tmp_artifacts/roc_curve.png'
        plot_confusion_matrix(y_test, y_pred, cm_path)
        plot_roc_curve(y_test, y_prob, roc_auc, roc_path)
        mlflow.log_artifact(cm_path,  artifact_path='plots')
        mlflow.log_artifact(roc_path, artifact_path='plots')

        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=['No Disease', 'Disease'],
            output_dict=True
        )
        cr_path = 'tmp_artifacts/classification_report.json'
        with open(cr_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(cr_path, artifact_path='reports')

        run_id = run.info.run_id
        print(f'\nRun ID  : {run_id}')
        print(f'Accuracy: {acc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'ROC AUC : {roc_auc:.4f}')
        print('\nTraining selesai!')


if __name__ == '__main__':
    main()
