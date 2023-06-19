from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from dataclasses import dataclass
from pycountry_convert import country_alpha2_to_continent_code

import os
import sys
import pandas as pd
import numpy as np
import pickle
import pycountry

from pathlib import Path
original_path = sys.path.copy()
sys.path.append(str(Path(__file__).parent.parent))
from src.logger import logging
from src.exception import CustomException
# from src.utils import save_object
sys.path = original_path

def get_continent(country_code):
    try:
        country = pycountry.countries.get(alpha_2=country_code)
        if country:
            return country.continent.alpha_2
    except Exception as e:
            raise CustomException(e, sys.exc_info())

@dataclass 
class Data_preprocessing_paths:
    df_path = "data/processed/input_data.csv"
    train_path = "data/processed/train_data.csv"
    test_path = "data/processed/test_data.csv"
    preprocessor_path = "data/processed/preprocessing.pkl"
    
class Data_processing:
    def __init__(self):
        data_preprocessing_paths_obj = Data_preprocessing_paths()
        self.df_path = data_preprocessing_paths_obj.df_path
        self.train_path = data_preprocessing_paths_obj.train_path
        self.test_path = data_preprocessing_paths_obj.test_path
        self.preprocessing_path = data_preprocessing_paths_obj.preprocessor_path
    
    def data_transformation(self):
        try:
            logging.info("Data Preprocessing started")
            train_df = pd.read_csv(self.train_path)

            target_column = "success"
            input_columns = train_df.columns.drop(target_column)

            duration_pipeline = Pipeline([
                ('duration', FunctionTransformer(lambda X: np.abs((pd.to_datetime(X['endDate'], format='%d-%m-%Y') - pd.to_datetime(X['startDate'], format='%d-%m-%Y')).dt.days), validate=False))
                ])

            platform_pipeline = Pipeline([
                ('platform_Ethereum', FunctionTransformer(lambda X: X.isin(['ETH', 'Ethererum', 'Ethereum', 'Ethereum, Waves', 'Etherum']).astype(int), validate=False))
                ])

            brand_slogan_pipeline = Pipeline([
                ('brandSlogan_score', FunctionTransformer(lambda X: X.apply(lambda x: TextBlob(str(x)).sentiment.polarity), validate=False))
                ])

            # country_region_pipeline = Pipeline([
            #     ('replace', FunctionTransformer(lambda X: X.applymap(get_continent))),
            #     ('one_hot_encoding', ColumnTransformer([
            #         ('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore'), ['countryRegion'])
            #     ]))
            # ])


            numerical_vars = train_df[input_columns].select_dtypes(include=['float64', 'int64']).columns.tolist()

            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
        
            preprocessor_transformers = [
                ('numerical', numerical_pipeline, numerical_vars),
                ('duration', duration_pipeline, ['startDate', 'endDate']),
                # ('country_region', country_region_pipeline, ['countryRegion']),
                ('platform', platform_pipeline, ['platform']),
                ('brand_slogan', brand_slogan_pipeline, ['brandSlogan'])
            ]

            preprocessor = ColumnTransformer(preprocessor_transformers, remainder='drop')

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys.exc_info())
    
    def data_preprocessing(self):
        try:
            df = pd.read_csv(self.df_path)

            logging.info("Reading data is completed")

            train_df, test_df = train_test_split(df, test_size=0.2, random_state= 42)

            train_df.to_csv(self.train_path, index=False)
            test_df.to_csv(self.test_path, index=False)
            
            logging.info("Data Spliting is completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.data_transformation()

            logging.info("Applying label encoder on training dataframe and testing dataframe for target column.")

            target_column = "success"

            label_encoder = LabelEncoder()
            train_df[target_column] = label_encoder.fit_transform(train_df[target_column])
            test_df[target_column] = label_encoder.transform(test_df[target_column])

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            dir_path = os.path.dirname(self.preprocessing_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(self.preprocessing_path, "wb") as file_obj:
                pickle.dump(preprocessing_obj, file_obj)
        
            logging.info("Saved preprocessing object.")

            return input_feature_train_arr, target_feature_train_df, input_feature_test_arr, target_feature_test_df, self.preprocessing_path

        except CustomException as e:
            new_exception = CustomException(e.error_message, sys)
            raise new_exception
