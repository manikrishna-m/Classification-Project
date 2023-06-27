import os
import sys
import pickle
import joblib
from dataclasses import dataclass
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

original_path = sys.path.copy()
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.exception import CustomException
from src.logger import logging

sys.path = original_path


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = "data/processed/model.pkl"


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.model_path = self.model_trainer_config.trained_model_file_path

    def initiate_model_trainer(self, train_input_array, train_target_array, test_input_array, test_target_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_input_array, train_target_array, test_input_array, test_target_array
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }
            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "splitter": ["best", "random"],
                    "max_features": ["sqrt", "log2"],
                },
                "Random Forest": {
                    "criterion": ["gini", "entropy"],
                    "max_features": ["sqrt", "log2", None],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    "loss": ["deviance", "exponential"],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "max_features": ["auto", "sqrt", "log2"],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Logistic Regression": {
                    "penalty": ["l2"],
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                },
                "XGBClassifier": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Classifier": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Classifier": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            dir_path = os.path.dirname(self.model_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(self.model_path, "wb") as file_obj:
                joblib.dump(best_model, file_obj)

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy

        except Exception as e:
            raise CustomException(str(e), sys)



def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            params = param[model_name]

            gs = GridSearchCV(model, params, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            test_model_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(str(e), sys)

