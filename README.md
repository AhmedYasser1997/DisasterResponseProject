# Disaster Response Pipeline Project

## Introduction
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The purpose of the project is to build a model for an API that classifies disaster messages that were created during a disaster into 36 categories to help in aid efforts. This would be of great help for some disaster relief agencies as a worker can input a message and get classification results in several categories so to have an idea what kind of help is needed: "medical", "water", "shelter", etc.. \
The Project is divided in the following Sections:

* Data Processing: Making an ETL Pipeline to extract data from source, clean data and save them in a database
* A Machine Learning Pipeline to train a model able to classify text message in categories
* A web app that displays some visualizations about the data and predict the category of the disaster given an input message.

## Files Description
### Data Folder
**disaster_messages.csv, disaster_categories.csv:** CSV files that contain real messages that were sent during disaster events and categories datasets in csv format. \
**process_data.py:** Python script that loads the messages and categories datasets, merges the two datasets ,cleans the data and stores it in a SQLite database
### Models Folder
**ML Pipeline Preparation.ipynb:** Jupyter notebook for building a machine learning pipeline and tuning it using GridSearchCV \
**train_classifier.py:** Python script that loads the data from the database, splits the dataset into training and test sets, builds a text processing and machine learning pipeline and exports the final model as a pickle file \
### App Folder
**run.py:** Python script for running the web app \
**go.html and master.html:** templates folder that contains 2 HTML files for the app front-end

## Running the project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/

## Summary 
The final output of the project is an interactive web app that first shows some data visualizations about the dataset and also takes a message from the user as an input and then classifies it.

![Capture1](https://github.com/AhmedYasser1997/DisasterResponseProject/blob/master/Capture1.PNG)

![Capture2](https://github.com/AhmedYasser1997/DisasterResponseProject/blob/master/Capture2.PNG)

