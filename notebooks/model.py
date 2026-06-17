import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import TunedThresholdClassifierCV, TimeSeriesSplit, RandomizedSearchCV

mlflow.set_tracking_uri("file:mlruns")

def plot_importance(model, feature_names, title_suffix=""):
    """
    plotting Feature importance, and upload to MLflow automatically
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values()

    fig, ax = plt.subplots(figsize=(10, 15))
    importances.plot(kind='barh', ax=ax)

    title = f'Feature Importance ({title_suffix})' if title_suffix else "Feature Importance"
    ax.set_title(title)
    fig.tight_layout()

    mlflow.log_figure(fig, f'feature_importance_{title_suffix}.png' if title_suffix else 'feature_importance.png')

    plt.show()
    plt.close(fig)

#================================================================

def random_forest_model(X_train, y_train, X_test, y_test, feature_names, 
                        experiment_run_name='RF_Model',n_estimators=200, 
                        max_depth=12, n_jobs=-1, threshold=0.5, class_weight=None):
    """
    A RandomForest model with automatical recording by 
    default param: 
        n_estimators=200, max_depth=12, n_job=-1
    """ 
    mlflow.end_run()
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name=experiment_run_name) as run:
        # record data processing method as tag
        mlflow.set_tag("data_process_method", experiment_run_name)

        # model training 
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth = max_depth,
            random_state=42,
            n_jobs = n_jobs,
            class_weight=class_weight)
        
        model.fit(X_train, y_train)

        # evaluation and recording Metrics
        y_proba = model.predict_proba(X_test)[:, 1]
        #y_pred = model.predict(X_test)
        y_pred = (y_proba > threshold).astype(int) # threshold should be 0.05 or something
        
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_param('threshold', threshold)
        print("Model training completed!")
        print(classification_report(y_test, y_pred))

        cm =confusion_matrix(y_test, y_pred, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        fig, ax =plt.subplots(figsize=(6,6))
        disp.plot(ax=ax,cmap=plt.cm.Blues)
        plt.title(f'Normalized Confusion Matrix, threshold:{threshold}')
        # upload figure to mlflow
        plot_path = f'confusion_matrix_{threshold}.png' # to produce the real confusion matrix plot to cover the mlflow autolog's(threshold=0.5 default) 
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
        
       
        plot_importance(model, feature_names, title_suffix=experiment_run_name)
        
        return model
    
#===============================================

def random_forest_model_search(X_train, y_train, X_test, y_test, feature_names, 
                        experiment_run_name='RF_Model',n_estimators=200, 
                        max_depth=12, n_jobs=-1, threshold=0.5, class_weight=None):
    """
    A RandomForest model with automatical recording by 
    default param: 
        n_estimators=200, max_depth=12, n_job=-1, threshold=0.5, class_weight=None
    
    """ 
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name=experiment_run_name) as run:
        # record data processing method as tag
        mlflow.set_tag("data_process_method", experiment_run_name)

        #set parameter
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth':[10, 15, 20],
            'min_samples_split': [2,5,10],
            'min_samples_leaf':[1,2,4],
            'max_features':['sqrt', 'log2'],
        }

        # model training 
        rf_model = RandomForestClassifier(
            random_state=42,
            n_jobs = n_jobs,
            class_weight=class_weight)
        
        tscv = TimeSeriesSplit(n_splits=2)
        search = RandomizedSearchCV(
            rf_model,
            param_grid,
            n_iter=12,
            scoring='average_precision',
            cv=tscv,
            verbose=2,
            n_jobs=n_jobs
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        print("Best params:", search.best_params_)
        # evaluation and recording Metrics
        y_proba = best_model.predict_proba(X_test)[:, 1]
        #y_pred = model.predict(X_test)
        y_pred = (y_proba > threshold).astype(int) # threshold should be 0.05 or something
        
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_param('threshold', threshold)
        print("Model training completed!")
        print(classification_report(y_test, y_pred))

        cm =confusion_matrix(y_test, y_pred, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        fig, ax =plt.subplots(figsize=(6,6))
        disp.plot(ax=ax,cmap=plt.cm.Blues)
        plt.title(f'Normalized Confusion Matrix, threshold:{threshold}')
        # upload figure to mlflow
        plot_path = f'confusion_matrix_{threshold}.png' # to produce the real confusion matrix plot to cover the mlflow autolog's(threshold=0.5 default) 
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
        
       
        plot_importance(best_model, feature_names, title_suffix=experiment_run_name)
        
        return best_model
    
