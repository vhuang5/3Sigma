import pandas as pd
import numpy as np
import constants
import tensorflow as tf

def csv_data(filepath):
    """
    Reads in csv to pandas df
    :param filepath: path to csv file
    :returns: pd df from the csv file
    """

    csv_data = pd.read_csv(filepath, parse_dates=['Date'])
    return csv_data

def sort_data(stock_filepath, commodities_filepath):
    """
    Creates dataframes and turns them into sorted dataframes of equal lengths
    :param filepaths: paths to csv files
    :returns: stock data and commodities data to use for the model
    """
    #get stock data
    stock_data = csv_data(stock_filepath)
    commodities_data = csv_data(commodities_filepath)

    stock_data = stock_data.dropna()
    commodities_data = commodities_data.dropna()

    stock_data = stock_data.sort_values(by=['Date'])
    commodities_data = commodities_data.sort_values(by=['Date'])

    stock_dates = stock_data['Date'].tolist()
    commodities_dates = commodities_data['Date'].tolist()
    stock_dates_array = np.array(stock_dates)
    commodities_dates_array = np.array(commodities_dates)

    shared_dates = np.intersect1d(stock_dates_array, commodities_dates_array)

    stonk = stock_data[stock_data['Date'].isin(shared_dates)]
    commie = commodities_data[commodities_data['Date'].isin(shared_dates)]

    return stonk, commie

def drop_column(stock_data, commodities_data, column_name):
    """
    Drops specified column in both stock and commodities dataframe
    :param stock_data: dataframe containing historical SPY data
    :param commodities_data: dataframe containing Invesco DBC ETF data
    :param column_name: specific column to drop
    :returns: updated stock data and commodities data
    """
    updated_stocks = stock_data.drop(columns = [column_name])
    updated_commodities = commodities_data.drop(columns = [column_name])

    return updated_stocks, updated_commodities

def modify_historical(stock_data, commodities_data, start_date, end_date):
    """
    Updates both stock and commodities dataframe to include only desired historical range
    :param stock_data: dataframe containing historical SPY data
    :param commodities_data: dataframe containing historical Invesco DBC ETF data
    :param start_date: string that represents specific date to have as first data point (inclusive)
    :param end_date: string that represents specific date to have as last data point (inclusive)
    :returns: updated stock data and commodities data
    """
    filtered_stock = stock_data.loc[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)]
    filtered_commodities = commodities_data.loc[(commodities_data['Date'] >= start_date) & (commodities_data['Date'] <= end_date)]
    
    return filtered_stock, filtered_commodities

def normalize(stock_data, commodities_data):
    """
    Performs Min-Max Normalization on each column of SPY and Invesco DBC dataframes
    :param stock_data: dataframe containing SPY data
    :param commodities_data: dataframe containing Invesco DBC ETF data
    :returns: updated stock data and commodities data
    """
    stock_copy = stock_data.copy()
    for stock_column in stock_data.columns:
        max_val = stock_data[stock_column].max()
        min_val = stock_data[stock_column].min()
        stock_copy[stock_column] = (stock_data[stock_column] - min_val) / (max_val - min_val)
    
    commodity_copy = commodities_data.copy()
    for commodity_column in commodities_data.columns:
        max_val = commodities_data[commodity_column].max()
        min_val = commodities_data[commodity_column].min()
        commodity_copy[commodity_column] = (commodities_data[commodity_column] - min_val) / (max_val - min_val)
    
    return stock_copy, commodity_copy



def get_data(stock_filepath, commodities_filepath, train_start_date, train_end_date, test_start_date, test_end_date):
    """
    Updates both stock and commodities dataframe to include only desired historical range
    :param stock_filepath: filepath to SPY .CSV file
    :param commodities_filepath: filepath to Invesco DBC .CSV file
    :param train_start_date: string that represents specific date to have as first data point (inclusive) for training
    :param train_end_date: string that represents specific date to have as last data point (inclusive) for training
    :param test_start_date: string that represents specific date to have as first data point (inclusive) for testing
    :param test_end_date: string that represents specific date to have as last data point (inclusive) for testing
    :returns: 
        train data which corresponds to a 2-D numpy array where each array corresponds to a daily measurement of Invesco DBC (Open, High, Low, Close, Volume)
        train labels which corresponds to a 2-D numpy array where each array corresponds to daily measurement of SPY (Close)
        test data which corresponds to a 2-D numpy array where each array corresponds to a daily measurement of Invesco DBC (Open, High, Low, Close, Volume)
        test labels which corresponds to a 2-D numpy array where each array corresponds to daily measurement of SPY (Close)
    """

    stock_data, commodities_data = sort_data(stock_filepath, commodities_filepath)

    updated_stock_data, updated_commodities_data = drop_column(stock_data, commodities_data, 'Adj Close')

    train_stocks, train_commodities = modify_historical(updated_stock_data, updated_commodities_data, train_start_date, train_end_date)
    test_stocks, test_commodities = modify_historical(updated_stock_data, updated_commodities_data, test_start_date, test_end_date)

    norm_train_stocks, norm_train_commodities = normalize(train_stocks, train_commodities)
    norm_test_stocks, norm_test_commodities = normalize(test_stocks, test_commodities)

    np_train_stocks = norm_train_stocks.to_numpy()
    np_train_commodities = norm_train_commodities.to_numpy()
    np_test_stocks = norm_test_stocks.to_numpy()
    np_test_commodities = norm_test_commodities.to_numpy()

    train_data = tf.convert_to_tensor(np_train_commodities[:, 1:], dtype=tf.float32)
    train_labels = tf.convert_to_tensor(np_train_stocks[:, -2:-1], dtype=tf.float32)
    test_data = tf.convert_to_tensor(np_test_commodities[:, 1:], dtype=tf.float32)
    test_labels = tf.convert_to_tensor(np_test_stocks[:, -2:-1], dtype=tf.float32)

    return train_data, train_labels, test_data, test_labels

