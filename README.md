# Disaster Response App - Using ML Pipeline 

For the purpose of disaster management, disaster messages are collected and categorized . An important aspect of disaster management is the choice of the disaster categories which allow to take appropriate measures to manage the situstion. In case of accident, fire or other immediate threats, a reaction is expected within a few minutes. 

**How could AI and Machine Learning support disaster management in this context?**

A disaster response App using Machine Learning to classify disaster messages could be the answer to the question above. In this solution, a multi-class Machine Learning classifier could classify the disaster messages using NLP techniques. The outcoume of the classification would be made available to the disaster management office, which will take the appropriate measures to manage the situation.The solution proposed in this work contains three parts.

- **ETL Pipeline for data preparation (ETL-Pipeline-Preparation.ipynb)**

The ETL pipeline will prepare the data to make it clean for machine learning. The data are read from csv-files, tranformed and stored in a database-file.

- **Multi-Classes Machine Learning Pipeline (Disaster-Response-ML-Pipeline.ipynb)**

The multi-classes ML pipeline will read the disaster messages and thier categories from database-file mentionned above. An ML model will be built, trainned and stored on the local filesystem.

- **Flask web-App for the categorization of disaster messages (./app/run.py)**

The Flask web-app will load the ML model from the filesystem and the disaster messages from the database created with the ETL pipeline. It will offer a functionality to select a message and visualize the classes of the message instantly. This information could then be used by disaster management office. 


The [disaster messages](https://www.kaggle.com/davidshahshankhar/disasterresponsepipeline) dataset supporting this work is freely available on kaggle.com. It consists of 2 csv-files:

- messages.csv: file containing disater messages
- categories.csv: file containing different categories of disaster 


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
  - templates/master.html  # main page of web app
  - templates/go.html  # classification result page of web app
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



