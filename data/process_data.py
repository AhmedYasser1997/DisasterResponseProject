import sys
import pandas as pd
import sqlite3
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load Data from the csv files
    
    Arguments:
        messages_filepath -> Path to the messages csv file 
        categories_filepath -> Path to categories csv file 
    Output:
        df -> A dataframe that joins both files based on the id
    """
    
    # Loading both csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Concatenating both dataframes
    df = pd.concat([messages, categories],axis=1, join='inner')
    return df

def clean_data(df):
    """
    Cleaning the dataframe through one hot encoding and dropping unnecessary columns
    
    Arguments:
        df-> The dataframe to be cleaned
    
    Output:
        df -> Dataframe ready to be used by the ML model
    """
    # create a dataframe of the 36 individual category columns
    # We'll split the values in the categories column on the ; character so that each value becomes a separate column
    # Then we'll use the first row of categories dataframe to create column names for the categories data.
    categories = df.categories.str.split(pat=';', expand=True)
    row = categories.loc[0]

    # The column name will be all but the last 2 characters. For example 'water-0' will be 'water' 
    category_colnames = row.apply(lambda x: x[0:-2]).values
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Now, we convert category values to just numbers 0 or 1 
    # by iterating through the category columns to keep only the last character of each string (the 1 or 0). 
    # For example, related-0 becomes 0.
    
    for column in categories:
        categories[column] = categories[column].astype(str).str.split("-").str.get(1)
        # Set column to be of numeric type
        categories[column] = pd.to_numeric(categories[column])
        
    # Now, we drop the categories column from the df dataframe since it is no longer needed and replace it with the new category columns.
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1, join="outer")
    
    # Finally, we can do some cleaning
    # We drop the ducplicates in the data
    df.drop_duplicates(inplace=True)
    
    # Original column has over 50% of it missing values. We won't need anyways as the message column has no missing values and the values   are all in English
    df.drop(columns=['original'], inplace=True)
    
    # related column has max value of 2 and when this happens, all other categories are 0, so therefore it may not be related.
    # Also the rows with related value 2 are only 188 rows and the data is over 26000 rows. 
    # So we can safely remove those rows.
    df = df.query("related!=2")
    
    # And since data is not numeric, there is no outliers or distributions to check
    return df

def save_data(df, database_filename):
    """
    Save the cleaned & processed dataframe into a table in a database
    
    Arguments:
        df-> The cleaned & processed dataframe
        database_filename -> Path to categories csv file 
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # Table will be named messages
    df.to_sql('messages', engine, index=False, if_exists='replace')   


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()