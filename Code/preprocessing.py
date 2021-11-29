import pandas as pd
import numpy as np
import constants

def csv_data(filepath):
    """
    Reads in csv to pandas df
    :param filepath: path to csv file
    :returns: pd df from the csv file
    """

    csv_data = pd.read_csv(filepath)
    return csv_data

def join_commodities(commodities_dict):
    pass

def get_data(stock_filepath, commodities_filepaths):
    """
    Creates dataframes and turns them into usable matrices for the model
    :param filepaths: paths to csv files
    :returns: stock data and commodities data to use for the model
    """
    #get stock data
    stock_data = csv_data(stock_filepath)

    #get commodities data
    commodities_dict = {}
    for key in commodities_filepaths:
        commodities_dict[key] = csv_data(commodities_filepaths.get(key))
    
    commodities = join_commodities(commodities_dict)
    
    return stock_data, commodities
