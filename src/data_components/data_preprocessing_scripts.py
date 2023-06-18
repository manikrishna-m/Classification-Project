from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from dataclasses import dataclass

import os
import sys
import pandas as pd
import numpy as np
import pickle

from pathlib import Path
original_path = sys.path.copy()
sys.path.append(str(Path(__file__).parent.parent))
from src.logger import logging
from src.exception import CustomException
# from src.utils import save_object
sys.path = original_path

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

            # Exclude target column
            target_column = "success"
            input_columns = train_df.columns.drop(target_column)

            numerical_vars = train_df[input_columns].select_dtypes(include=['float64', 'int64']).columns.tolist()
            categorical_vars = train_df[input_columns].select_dtypes(include=['object']).columns.tolist()

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
            raise CustomException(e, sys.exc_info())

    # def data_transformation(self):
    #     try:
    #         logging.info("Data Preprocessing started")
    #         train_df = pd.read_csv(self.train_path)

    #         # Exclude target column
    #         target_column = "success"
    #         input_columns = train_df.columns.drop(target_column)

    #         duration_pipeline = Pipeline([
    #             ('duration', FunctionTransformer(lambda X: np.abs((pd.to_datetime(X['endDate'], format='%d-%m-%Y') - pd.to_datetime(X['startDate'], format='%d-%m-%Y')).dt.days), validate=False))
    #         ])

    #         country_pipeline = Pipeline([
    #             ('lowercase', FunctionTransformer(lambda X: X.str.lower())),
    #             ('replace', FunctionTransformer(lambda X: X.replace(['curacao', 'curaçao'], 'curacao')))
    #         ])

    #         country_region_pipeline = Pipeline([
    #             ('dummies', FunctionTransformer(lambda X: pd.get_dummies(X['countryRegion']), validate=False)),
    #             ('fillna', SimpleImputer(strategy='constant', fill_value=0))
    #         ])

    #         platform_pipeline = Pipeline([
    #             ('platform_Ethereum', FunctionTransformer(lambda X: X.isin(['ETH', 'Ethererum', 'Ethereum', 'Ethereum, Waves', 'Etherum']).astype(int), validate=False))
    #         ])

    #         brand_slogan_pipeline = Pipeline([
    #             ('brandSlogan_score', FunctionTransformer(lambda X: X.apply(lambda x: TextBlob(str(x)).sentiment.polarity), validate=False))
    #         ])

    #         numerical_vars = train_df[input_columns].select_dtypes(include=['float64', 'int64']).columns.tolist()
    #         categorical_vars = train_df[input_columns].select_dtypes(include=['object']).columns.tolist()

    #         numerical_pipeline = Pipeline([
    #             ('imputer', SimpleImputer(strategy='median')),
    #             ('scaler', StandardScaler())
    #         ])

    #         categorical_pipeline = Pipeline([
    #             ('imputer', SimpleImputer(strategy='most_frequent')),
    #         ])

    #         preprocessor_transformers = [
    #             ('numerical', numerical_pipeline, numerical_vars),
    #             ('categorical', categorical_pipeline, categorical_vars),
    #             ('duration', duration_pipeline, ['startDate', 'endDate']),
    #             ('country', country_pipeline, ['countryRegion']),
    #             ('country_region', country_region_pipeline, ['countryRegion']),
    #             ('platform', platform_pipeline, ['platform']),
    #             ('brand_slogan', brand_slogan_pipeline, ['brandSlogan'])
    #         ]

    #         preprocessor = ColumnTransformer(preprocessor_transformers, remainder='drop')

    #         return preprocessor

    #     except Exception as e:
    #         raise CustomException(e, sys.exc_info())


    
    
    def data_preprocessing(self):
        try:
            df = pd.read_csv(self.df_path)

            logging.info("Reading data is completed")

            logging.info("Data Cleaning is started")
            
            df['priceUSD'] = np.where(df['priceUSD'] == 0, df['priceUSD'].median(), df['priceUSD'])

            df['duration'] = np.abs((pd.to_datetime(df['endDate'], format='%d-%m-%Y') - pd.to_datetime(df['startDate'], format='%d-%m-%Y')).dt.days)
            df = df.drop(['startDate','endDate'], axis=1)

            df['countryRegion'] = df['countryRegion'].str.lower().replace(['curacao', 'curaçao'], 'curacao')
            top_10_countries = df['countryRegion'].value_counts().head(10).index.tolist()
            filtered_df = df[df['countryRegion'].isin(top_10_countries)]
            df = pd.concat([df, pd.get_dummies(filtered_df['countryRegion'])], axis=1)
            countries_to_fill = ['cayman islands', 'estonia','germany', 'malta', 'netherlands', 'russia', 'singapore', 'switzerland','uk', 'usa',]
            df[countries_to_fill] = df[countries_to_fill].fillna(0)
            df = df.drop(['countryRegion'], axis=1)

            df['platform_Ethereum'] = np.where(df['platform'].isin(['ETH', 'Ethererum', 'Ethereum', 'Ethereum, Waves', 'Etherum']), 1, 0)
            df = df.drop(['platform'], axis=1)

            df['brandSlogan_score'] = df['brandSlogan'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            df = df.drop(['brandSlogan'], axis=1)

            logging.info("Data Cleaning is completed")
            
            train_df, test_df = train_test_split(df, test_size=0.2, random_state= 42)
            train_df.to_csv(self.train_path, index=False)
            test_df.to_csv(self.test_path, index=False)
            
            logging.info("Data Spliting is completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.data_transformation()

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
