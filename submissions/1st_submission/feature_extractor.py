import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, maxabs_scale
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from category_encoders.m_estimate import MEstimateEncoder
from sklearn.preprocessing import OrdinalEncoder


class FeatureExtractor(object):
    
    y = None
    
    def init(self):
        pass
    
    def fit(self, X_df, y_array):
        
        self.y = y_array
        
        path = os.path.dirname(__file__)
        award = pd.read_csv(os.path.join(path, 'award_notices_RAMP.csv.zip'),compression='zip', low_memory=False)
        #date = pd.to_datetime(award['Publication_date'], format='%Y-%m-%d')
        #award['Year'] = date.dt.year
        award['Name_processed'] = award['incumbent_name'].str.lower()
        award['Name_processed'] = award['Name_processed'].str.replace('[^\w]','')
        award_features = award.groupby(['Name_processed'])['amount'].agg(['count','sum'])
        
        def zipcodes(X):
            zipcode_nums = pd.to_numeric(X['Zipcode'], errors='coerce')
            # Reduce zipcode to departement
            zipcode_nums = zipcode_nums.apply(lambda x : np.trunc(x/1000) if x > 100 else x)
            return zipcode_nums.values[:, np.newaxis]
        
        zipcode_transformer = FunctionTransformer(zipcodes, validate=False)
        
        numeric_transformer = Pipeline(steps=[('impute', SimpleImputer(strategy='median'))])

        def process_date(X):
            date = pd.to_datetime(X['Fiscal_year_end_date'], format='%Y-%m-%d')
            return date.dt.month.values[:, np.newaxis]

        date_transformer = FunctionTransformer(process_date, validate=False)

        def number_years(X):
            X['Name'] = X['Name'].fillna('NaN')
            n_years = X.groupby('Name').size()
            min_year = X.groupby('Name')['Year'].min()
            df = pd.concat([n_years, min_year], axis=1, join='inner')
            df.columns = ['Age', 'creation_date']
            X = X.set_index('Name')
            X = pd.merge(X, df, how='left', left_index=True,right_index=True)
            X['anc'] = X['Year'] - X['creation_date']
            X = X.reset_index()
            return X['anc'].values[:,np.newaxis]
          
        year_transformer = FunctionTransformer(number_years, validate=False)
        
        def process_APE(X):
            APE = X['Activity_code (APE)'].str[:2]
            return pd.to_numeric(APE).values[:, np.newaxis]

        APE_transformer = FunctionTransformer(process_APE, validate=False)
        
        def merge_naive(X):
            X['Name'] = X['Name'].str.lower()
            X['Name'] = X['Name'].str.replace('[^\w]','')
            df = pd.merge(X, award_features, left_on='Name',
                          right_on='Name_processed', how='left')
            return df[['count','sum']]

        merge_transformer = FunctionTransformer(merge_naive, validate=False)
            
        def head_imput(X):

            X['Name'] = X['Name'].fillna('NaN')
            map_h = abs(X.groupby('Name')['Headcount'].median()).to_dict()
            X['Headcount'] = X['Name'].map(map_h)
            return X['Headcount'].values[:, np.newaxis]

        head_transformer = FunctionTransformer(head_imput, validate=False)

        
        num_cols = ['Legal_ID','Year']
        head_cols = ['Headcount','Name']
        zipcode_col = ['Zipcode']
        date_cols = ['Fiscal_year_end_date']
        APE_col = ['Activity_code (APE)']
        merge_col = ['Name']
        drop_cols = ['Address', 'City','Fiscal_year_duration_in_months', 'Name']
        year_col = ['Year', 'Name']
        MEstimate_col = ['Activity_code (APE)','Zipcode']
        
        preprocessor = ColumnTransformer(transformers=[
                                                       ('zipcode', make_pipeline(zipcode_transformer, SimpleImputer(strategy='median')), zipcode_col),
                                                       ('num', numeric_transformer, num_cols),
                                                       ('head',head_transformer, head_cols),
                                                       ('APE', make_pipeline(APE_transformer, SimpleImputer(strategy='median')), APE_col),
                                                       ('merge', make_pipeline(merge_transformer, SimpleImputer(strategy='median')), merge_col),
                                                       ('target',TargetEncoder(cols=MEstimate_col), MEstimate_col),
                                                       ('year', make_pipeline(year_transformer, SimpleImputer(strategy='median')), year_col),
                                                       ('drop cols', 'drop', drop_cols)
                                                       ])
            
        self.preprocessor = preprocessor
        self.preprocessor.fit(X_df, y_array)
        return self
                                                       
    def transform(self, X_df):
        return self.preprocessor.transform(X_df)
