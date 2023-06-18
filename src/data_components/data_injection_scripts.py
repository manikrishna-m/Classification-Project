from src.logger import logging
from src.exception import CustomException

import os
import sys
from pathlib import Path
original_path = sys.path.copy()
sys.path.append(str(Path(__file__).parent.parent))

from data_components.model_evaluation_scripts import ModelTrainer
from data_components.data_preprocessing_scripts import Data_processing

import pandas as pd
import numpy as np

from dataclasses import dataclass

sys.path = original_path

@dataclass
class Data_injection_paths:
    input_data_path: str = "data/input/input_data.csv"
    traintest_data_path: str = "data/processed/input_data.csv"

class Data_injection:
    def __init__(self):
        data_inj_path_obj = Data_injection_paths()
        self.input_path = data_inj_path_obj.input_data_path
        self.train_test_path = data_inj_path_obj.traintest_data_path
        
    def load_data(self):
        try:
            logging.info("Data Injection started")

            input_data = pd.read_csv(self.input_path)

            logging.info("Data Injection Completed")

            input_data.to_csv(self.train_test_path, index=False)

            logging.info("Data is stored in processed folder")

        except Exception as e:
            logging.exception("Data Injection Exception: {}".format(e))
            raise CustomException(e, sys)

if __name__=="__main__":
    obj=Data_injection()
    obj.load_data()

    data_transformation=Data_processing()
    train_input_arr, train_target_arr, test_input_arr, test_target_arr, _ = data_transformation.data_preprocessing()

    # modeltrainer=ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_input_arr, train_target_arr, test_input_arr, test_target_arr))