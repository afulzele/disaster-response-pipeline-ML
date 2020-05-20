# Disaster Response Pipeline Project

This project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The initial dataset contains pre-labelled tweet and messages from real-life disaster. The aim of the project is to classify disaster response messages by building a Natural Language Processing tool that categorize messages.

The Project is divided in the following Sections:

1. ETL Pipeline to extract data from source, clean data and save them in a database.

2. Machine Learning Pipeline to train a model able to classify text message in categories

3. Web App to use model to classify the query in results in real time.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

**Note:** It takes sometime to load the page since counting the *Top 20 words* takes a little processing time. I need to make this asynchronous so that it doesn't affect the frontend.

### Screenshots:

#### Web App

![alt text](https://github.com/afulzele/disaster-response-pipeline/blob/master/Web-app.png)

#### Graphs

![alt text](https://github.com/afulzele/disaster-response-pipeline/blob/master/Distribution%20of%205%20popular%20categories.png)

![alt text](https://github.com/afulzele/disaster-response-pipeline/blob/master/20%20most%20common%20words.png)

![alt text](https://github.com/afulzele/disaster-response-pipeline/blob/master/Distribution%20of%20genre.png)

#### Query Result

![alt text](https://github.com/afulzele/disaster-response-pipeline/blob/master/Query-Category-Prediction-1.png)

![alt text](https://github.com/afulzele/disaster-response-pipeline/blob/master/Query-Category-Prediction-2.png)

#### Acknowledgement

* Udacity for providing this project.
* Figure Eight for providing messages dataset.
