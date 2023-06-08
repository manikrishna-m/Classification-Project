from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from textblob import TextBlob

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass

import pandas as pd
import numpy as np
import sys
import os

@dataclass 
class Data_preprocessing_paths:
    train_path = "data/processed/train_data.csv"
    test_path = "data/processed/test_data.csv"
    preprocessor_path = "data/processed/preprocessing.pkl"
    
class Data_processing:
    def __init__(self):
        data_preprocessing_paths_obj = Data_preprocessing_paths()
        self.train_path = data_preprocessing_paths_obj.train_path
        self.test_path = data_preprocessing_paths_obj.test_path
        self.preprocessing_path = data_preprocessing_paths_obj.preprocessor_path
    
    def data_transformation(self):
        try:
            
            logging.info("Data Preprocessing started")
            
            train_data = pd.read_csv(self.train_path)
            
            numerical_vars = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            categorical_vars = train_data.select_dtypes(include=['object']).columns.tolist()

            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer([
                ('numerical', numerical_pipeline, numerical_vars),
                ('categorical', categorical_pipeline, categorical_vars)
            ])

            return preprocessor 
        
        except Exception as e:
            raise CustomException(e,sys)
    
    
    def data_preprocessing(self):

        try:
            train_df=pd.read_csv(self.train_path)
            test_df=pd.read_csv(self.test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.data_transformation()

            target_column="success"

            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            label_encoder = LabelEncoder()
            target_feature_train_df['success'] = label_encoder.fit_transform(target_feature_train_df['success'])
            target_feature_test_df['success'] = label_encoder.fit_transform(target_feature_test_df['success'])

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.preprocessing_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.preprocessing_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        