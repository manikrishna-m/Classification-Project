import pandas as pd
import numpy as np
import sys
import os 
import pickle
import joblib

from pathlib import Path
original_path = sys.path.copy()
sys.path.append(str(Path(__file__).parent.parent))
from src.exception import CustomException
from src.logger import logging
from src.data_components.data_injection_scripts import Data_injection
from src.data_components.data_preprocessing_scripts import Data_processing
from src.data_components.model_evaluation_scripts import ModelTrainer
sys.path = original_path


class Predict_pipeline:
    def __init__(self):
        self.processor_path = 'data/processed/preprocessing.pkl'
        self.model_path = 'data/processed/model.pkl'

    def predict(self,data):
        try:
            with open(self.processor_path, "rb") as file_obj:
                processor = joblib.load(file_obj)

            with open(self.model_path, "rb") as file_obj:
                model = joblib.load(file_obj)

            return model.predict(processor.transform(data))
        
        except Exception as e:
            logging.exception("Data Injection Exception: {}".format(e))
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, brandSlogan,hasVideo,rating,priceUSD,countryRegion,startDate,endDate,teamSize,
                 hasGithub,hasReddit,platform,coinNum,minInvestment,distributedPercentage):
        self.brandSlogan = brandSlogan
        self.hasVideo = int(hasVideo)
        self.rating = int(rating)
        self.priceUSD = int(priceUSD)
        self.countryRegion = countryRegion
        self.startDate = startDate
        self.endDate = endDate
        self.teamSize = int(teamSize)
        self.hasGithub = int(hasGithub)
        self.hasReddit = int(hasReddit)
        self.platform = platform
        self.coinNum = int(coinNum)
        self.minInvestment = int(minInvestment)
        self.distributedPercentage = float(distributedPercentage)

    def data_dict(self):
        try:
            predict_df = {
                'ID': [1],
                'hasVideo': [self.hasVideo],
                'rating': [self.rating],
                'priceUSD': [self.priceUSD],
                'teamSize': [self.teamSize],
                'hasGithub': [self.hasGithub],
                'hasReddit': [self.hasReddit],
                'coinNum': [self.coinNum],
                'minInvestment': [self.minInvestment],
                'distributedPercentage': [self.distributedPercentage],
                'startDate': [self.startDate],
                'endDate': [self.endDate],
                'countryRegion': [self.countryRegion],
                'platform': [self.platform],
                'brandSlogan': [self.brandSlogan],
            }

            return pd.DataFrame.from_dict(predict_df, orient='columns')

        except Exception as e:
            logging.exception("Data Injection Exception: {}".format(e))
            raise CustomException(e, sys)







