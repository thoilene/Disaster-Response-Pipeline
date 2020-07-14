# Project: Disaster Response Pipeline

* This is a project in the context of the Nanodegree program "Data Science" 
* It intends to show the ability of the student to create and use ETL pipeline and Machine Learning (ML) pipeline in Data Engineering as well as to develop a web app with visualization (Software Engineering).

## Requirements

The following libraries have been used for this project under Anaconda distribution:

- Python 3.7.4
- joblib 0.13.2
- numpy 1.16.5
- pandas 0.25.1
- seaborn 0.9.0
- matplotlib 3.1.1
- sklearn 0.21.3
- sqlalchemy 1.3.9

## Project motivation

Disaster management is a challenging task as it could be difficult to identify the type of disaster in order to take the right mesures. It could be helpful for the operator receiving the disaster message to assign it to the right category. In this project a classifier has been developped as predictive model to determine the category of a disaster given a disaster message. A web app is developped to be used as front end solution.

## Files in the Repository

- Write A Data Science Blog Post.ipynb: Jupyter notebook containing the step of analysis and the code of the project
  
- app
  - templates
   - master.html  # main page of web app
   - go.html  # classification result page of web app
  - run.py  # Flask file that runs app

- data
  - disaster_categories.csv  # data to process 
  - disaster_messages.csv  # data to process
  - process_data.py
  - DisasterResponse.db   # database to save clean data to

- models
  - train_classifier.py 
  - classifier.pkl  # saved model 

- README.md

## How to run the Python scripts and web app

The project is structured in 3 modules.

### Data module

This module prepares the data using an ETL pipeline. The following command runs an ETL pipeline in order to clean data and store in database (from the project folder):

#### #python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv


### Model module

The model module creates, trains a model using an ML pipeline with a multi-output classifier and save it. The following command runs an ML pipeline in order to create, train a classifier and save it in a file (from the project folder):

#### #python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl 

### Web app module

The web app module offer a front end solution for the use of the classifier. The following command starts the web app from the app's directory:

#### #python run.py

After starting the web app - the app is accessible via a browser. Go to http://127.0.0.1:3001/ to this end.



