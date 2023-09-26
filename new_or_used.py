"""
Exercise description
--------------------

Description:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the markeplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution 
to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k_checked_v3.jsonlines` and a function to read that dataset in `build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. 
Additionally, you will have to choose an appropiate secondary metric and also elaborate an argument on why that metric was chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features, 
  the proposed secondary metric and the performance achieved on that metrics. 
  Optionally, you can deliver an EDA analysis with other formart like .ipynb



"""

import json
import re
import pandas as pd
from pandas import json_normalize
import numpy as np
import os
import datetime

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# You can safely assume that `build_dataset` is correctly implemented
def build_dataset():
    data = [json.loads(x) for x in open("MLA_100k_checked_v3.jsonlines")]
    target = lambda x: x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    for x in X_test:
        del x["condition"]
    return X_train, y_train, X_test, y_test

def pre_process(text):
    ''' funcion que elimina caracteres especiales en columnas '''
    text = str(text)                             # Converting texto into string
    text = re.sub(r'[^\w\s]', ' ', text)         # Remove all the special characters
    return text

def fechas(df, column_to_look):
    ''' Agrupa y cuenta los datos en diferentes periodos y los agrega al dataframe original '''
    column_to_look = str(column_to_look)
    df['datetime'] = pd.to_datetime(df[column_to_look],format='%Y-%m-%d').dt.date
    df['dia'] = pd.to_datetime(df[column_to_look], dayfirst=True).dt.day.astype('int')
    df['semana'] = pd.to_datetime(df[column_to_look], dayfirst=True).dt.isocalendar().week
    df['mes'] = pd.to_datetime(df[column_to_look], dayfirst=True).dt.month.astype('int')
    df['ano'] = pd.to_datetime(df[column_to_look], dayfirst=True).dt.year.astype('int')
    return df

def clean_flattern_json(df):
    ''' Función que aplana y limpia los datos aplicando la función de json_normalize y pre_process '''
    #1st normalization: flatten columns with json files
    dfp = json_normalize(df)
    #2nd normalization: flatten nested columns with json files
    nested_columns = ['non_mercado_pago_payment_methods', 'pictures']
    for col in nested_columns:
        df_col = json_normalize(df, record_path=col)
        columns_name = df_col.columns
        for ncols in columns_name:
            dfp[col+'.'+ncols] = df_col[ncols]
    #remove nested columns 
    dfp.drop(nested_columns, axis='columns', inplace=True)
    #3rd normalization: remove special characters from columns
    columns_with_special_characters = ['sub_status',
                                       'deal_ids',
                                       'variations',
                                       'attributes',
                                       'tags',
                                       'coverage_areas',
                                       'descriptions',
                                       'shipping.methods',
                                       'shipping.tags']
    for col in columns_with_special_characters:
        dfp[col] = dfp[col].apply(pre_process)
    #4th convert boolean columns into strings
    boolean_columns = ['accepts_mercadopago','automatic_relist','shipping.local_pick_up','shipping.free_shipping']
    for col in boolean_columns:
        dfp[col] = dfp[col].map({True: 'True', False: 'False'}) 
    return dfp

def select_columns(numerical_columns, categorical_columns, X_train_df, X_test_df):
        X_full_df = pd.concat([X_train_df, X_test_df], axis=0)
        X_full_num = X_full_df[numerical_columns]
        X_full_cat = X_full_df[categorical_columns]
        return X_full_num, X_full_cat

def vectorize_categorical_columns(X_full_cat):
        vectorizer = OneHotEncoder()
        X_full_categorical = vectorizer.fit_transform(X_full_cat).toarray()
        return X_full_categorical


if __name__ == "__main__":
    print("Loading dataset...")
    # Train and test data following sklearn naming conventions
    # X_train (X_test too) is a list of dicts with information about each item.
    # y_train (y_test too) contains the labels to be predicted (new or used).
    # The label of X_train[i] is y_train[i].
    # The label of X_test[i] is y_test[i].
    X_train, y_train, X_test, y_test = build_dataset()

    # Insert your code below this line:
    
    # Prepare and clean data
    print('Cleaning data! ')
    X_train_df = clean_flattern_json(X_train)    
    X_test_df = clean_flattern_json(X_test)
    
    # Select Columns to Train
    numerical_columns = ['stop_time',
                         'start_time',
                         'available_quantity',
                         'price',
                         'initial_quantity',
                         'sold_quantity']
    
    categorical_columns = ['listing_type_id', 
                           'buying_mode', 
                           'accepts_mercadopago', 
                           'currency_id', 
                           'automatic_relist', 
                           'status',
                           'seller_address.state.name',
                           'seller_address.city.name',
                           'shipping.local_pick_up', 
                           'shipping.free_shipping', 
                           'shipping.mode',
                           'non_mercado_pago_payment_methods.description',
                           'non_mercado_pago_payment_methods.type']
    
    print('Selecting features to train! ')
    X_full_num, X_full_cat = select_columns(numerical_columns, categorical_columns, X_train_df, X_test_df)
    
    print('Vectorizing categorical features! ')
    X_full_categorical = vectorize_categorical_columns(X_full_cat)
    X_full_numerical = X_full_num.to_numpy()
    X_full = np.concatenate((X_full_numerical, X_full_categorical), axis=1)

    print('Preparing data! ')
    X_train = X_full[:-10000]
    X_test = X_full[-10000:]

    print('Trainig model! This section can take several minutes')
    rf_model = RandomForestClassifier(n_estimators=300, random_state=0)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    print('Classification Report!')
    print(classification_report(y_test, y_pred))
    accuracy = 100*accuracy_score(y_test, y_pred)
    print('Accuracy: {:.2f}%'.format(accuracy))
    
    print('\nEnd of the Code! ')
