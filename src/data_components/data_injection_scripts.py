import sys
import pandas as pd

from dataclasses import dataclass
from pathlib import Path

original_path = sys.path.copy()
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.logger import logging
from src.exception import CustomException
from src.data_components.model_evaluation_scripts import ModelTrainer
from src.data_components.data_preprocessing_scripts import Data_processing

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
            logging.info("Reading input dataset file")

            input_data = pd.read_csv(self.input_path)

            input_data.to_csv(self.train_test_path, index=False)

            logging.info("Data is stored in processed folder")

        except Exception as e:
            logging.exception("Data Injection Exception: {}".format(e))
            raise CustomException(e, sys)

