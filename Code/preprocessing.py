import pandas as pd
import numpy as np
import constants

def csv_data(filepath):
    """
    Reads in csv to pandas df
    :param filepath: path to csv file
    :returns: pd df from the csv file
    """

    csv_data = pd.read_csv(filepath, parse_dates=["Date"])
    return csv_data

def join_commodities(commodities_dict):
    pass

def get_data(stock_filepath, commodities_filepath):
    """
    Creates dataframes and turns them into usable matrices for the model
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

    return stock_data, commodities_data

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

def modify_historical(stock_data, commodities_data, start_date, end_date)
    """
    Updates both stock and commodities dataframe to include only desired historical range
    :param stock_data: dataframe containing historical SPY data
    :param commodities_data: dataframe containing Invesco DBC ETF data
    :param start_date: string that represents specific date to have as first data point (inclusive)
    :param end_date: string that represents specific date to have as last data point (inclusive)
    :returns: updated stock data and commodities data
    """
    filtered_stock = stock_data.loc[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)]
    filtered_commodities = commodities_data.loc[(commodities_data['Date'] >= start_date) & (commodities_data['Date'] <= end_date)]
    
    return filtered_stock, filtered_commodities



def convert
