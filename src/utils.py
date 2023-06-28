import sys
from pathlib import Path

original_path = sys.path.copy()
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.logger import logging
from src.data_components.data_injection_scripts import Data_injection
from src.data_components.data_preprocessing_scripts import Data_processing
from src.data_components.model_evaluation_scripts import ModelTrainer

sys.path = original_path

if __name__=="__main__":
    logging.info("Data Injection is started")
    obj=Data_injection()
    obj.load_data()
    logging.info("Data Injection is finished")

    logging.info("Data Preprocessing pipelines is started")
    data_transformation=Data_processing()
    train_input_arr, train_target_arr, test_input_arr, test_target_arr, _ = data_transformation.data_preprocessing()
    logging.info("Data preproceesing pipelines is finised")

    logging.info("Model Building is started")
    modeltrainer=ModelTrainer()
    logging.info("Best model accuracy is", modeltrainer.initiate_model_trainer(train_input_arr, train_target_arr, test_input_arr, test_target_arr))
    logging.info("Model buiding is finished")
