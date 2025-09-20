import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        ''' This function is responsible for data transformation '''
        try:
            logging.info("Data Transformation initiated")
            # Define which columns should be ordinal-encoded and which should be scaled
            numeric_features = ['kms_run', 'times_viewed', 'broker_quote',
       'original_price', 'emi_starts_from', 'booking_down_pymnt','yr_mfr']
            categorical_columns = ['car_name', 'fuel_type', 'body_type', 'transmission', 
                                'source' , 'car_availability']
            ordinal_columns = ['car_rating','total_owners']

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            # Ordinal Pipeline
            ord_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('label_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                    ('scaler', StandardScaler())
                ]
            )


            

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numeric_features),
                    ('cat_pipeline', cat_pipeline, categorical_columns),
                    ('ord_pipeline', ord_pipeline, ordinal_columns)
                ]
            )

            logging.info(f"Categorical columns encoding completed")
            logging.info(f"Numerical columns scaling completed")

            return preprocessor
        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            

           
            logging.info("Obtaining preprocessor object")
            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'sale_price'
            drop_columns = ['sale_price','ad_created_on','make','model','rto','variant', 'registered_city', 'registered_state','city']

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Transforming using preprocessor object
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            print(input_feature_train_arr.shape, input_feature_test_arr.shape)
            print(target_feature_train_df.shape, target_feature_test_df.shape)
            # Combining transformed features and target variable into one array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Saving the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.info("Exception occurred in the initiate_data_transformation")
            raise CustomException(e, sys)
