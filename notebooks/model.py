import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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



def random_forest_model(X_train, y_train, X_test, y_test, feature_names, 
                        experiment_run_name='RF_Model',n_estimators=200, max_depth=12, n_jobs=-1):
    """
    A RandomForest model with automatical recording by 
    default param: 
        n_estimators=200, max_depth=12, n_job=-1
    """ 
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name=experiment_run_name) as run:
        # record data processing method as tag
        mlflow.set_tag("data_process_method", experiment_run_name)

        # model training 
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth = max_depth,
            random_state=42,
            n_jobs = n_jobs)
        
        model.fit(X_train, y_train)

        # evaluation and recording Metrics
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", acc)

        print("Model training completed!")
        print(classification_report(y_test, y_pred))

        plot_importance(model, feature_names, title_suffix=experiment_run_name)
        
        return model


