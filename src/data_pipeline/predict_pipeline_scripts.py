import pandas as pd
import numpy as np
import sys
import os 
import pickle

from pathlib import Path
original_path = sys.path.copy()
sys.path.append(str(Path(__file__).parent.parent))
from src.exception import CustomException
sys.path = original_path


class Predict_pipeline:
    def __init__(self):
        self.processor_path = 'data/processed/processor.pkl'
        self.model_path = 'data/processed/model.pkl'

    def predict(self,data):
        try:
            with open(self.processor_path, "rb") as file_obj:
                processor = pickle.load(file_obj)

            with open(self.model_path, "rb") as file_obj:
                model = pickle.load(file_obj)

            return model.predict(processor.tranform(data))
        except Exception as e:
            raise CustomException(str(e),sys)


class CustomData:
    def __init__(self, brandSlogan,hasVideo,rating,priceUSD,countryRegion,startDate,endDate,teamSize,
                 hasGithub,hasReddit,platform,coinNum,minInvestment,distributedPercentage):
        self.brandSlogan = brandSlogan
        self.hasVedio = hasVideo
        self.rating = rating
        self.priceUSD = priceUSD
        self.countryRegion = countryRegion
        self.startDate = startDate
        self.endDate = endDate
        self.teamSize = teamSize
        self.hasGithub = hasGithub
        self.hasReddit = hasReddit
        self.platform = platform
        self.coinNum = coinNum
        self.minInvestment = minInvestment
        self.distributedPercentage = distributedPercentage

    def data_dict(self):
        try:
            predict_df = {
                'brandSlogan': self.brandSlogan,
                'hasVedio': self.hasVedio,
                'rating': self.rating,
                'priceUSD': self.priceUSD,
                'countryRegion': self.countryRegion,
                'startDate': self.startDate,
                'endDate': self.endDate,
                'teamSize': self.teamSize,
                'hasGithub': self.hasGithub,
                'hasReddit': self.hasReddit,
                'platform': self.platform,
                'coinNum': self.coinNum,
                'minInvestment': self.minInvestment,
                'distributedPercentage': self.distributedPercentage
            }

            return pd.DataFrame(predict_df)
        
        except Exception as e:
            raise CustomException(str(e),sys)






