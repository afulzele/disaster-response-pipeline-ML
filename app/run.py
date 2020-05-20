import json
import plotly
import pandas as pd
import re

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine
from collections import Counter


app = Flask(__name__)


#-------------------- Function that uses NLP to convert sentences in tokens
stop_words = stopwords.words("english")

def tokenize(text):

    # We'll remove all the punctuations from a sentence
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Then we'll seperate words
    tokens = word_tokenize(text) 

    # Initilizing a lemmatize function    
    lemmatizer = WordNetLemmatizer()

    # Using list comprehension to get a list of lemmatized (root words) words which are not in the stopwords
    token = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return token    
    

# load data from SQL to dataframe
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disasterTable', engine)


# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    #-------------------- extract data needed for visuals

    # extracting name and values of genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extracting the top 5 common categories
    category_count = df.iloc[:,4:].sum().sort_values(ascending=False)[:5]
    category_names = list(category_count.index)
    
    # extracting the top 20 words extarcted using NLP
    wordfreq={}
    mess = df['message'].tolist()
    for m in mess:
        token = tokenize(m)
        for t in token:
            if t not in wordfreq.keys():
                wordfreq[t] = 1
            else:
                wordfreq[t] += 1
    
    word_counts = []
    word_names = []
    ar = dict(Counter(wordfreq).most_common(20))
    for i,v in ar.items():
        word_counts.append(v)
        word_names.append(i)
    
    # create visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts,
                    hole=.3,
                    textinfo='label+percent',
                    showlegend=False
                )
            ],

            'layout': {
                'title': {
                    'text': 'Distribution of Message Genres',
                    'font': dict(
                        family="Courier New, monospace",
                        size=24
                    )
                },
                'paper_bgcolor':'rgb(248, 243, 235)',
                'plot_bgcolor':'rgb(248, 243, 235)',
            }
        },
        {
            'data': [
                Bar(
                    x=word_names,
                    y=word_counts
                )
            ],

            'layout': {
                'title': {
                    'text': '20 Most common words',
                    'font': dict(
                        family="Courier New, monospace",
                        size=24
                    )
                },
                'paper_bgcolor':'rgb(248, 243, 235)',
                'plot_bgcolor':'rgb(248, 243, 235)',
                'yaxis': {
                    'title': {
                        'text': "Count",
                        'font': dict(
                            family="Courier New, monospace",
                            size=18,
                            color="#7f7f7f"
                        )
                    }
                },
                'xaxis': {
                    'title': {
                        'text': "Words",
                        'font': dict(
                            family="Courier New, monospace",
                            size=18,
                            color="#7f7f7f"
                        )
                    }
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_count
                )
            ],

            'layout': {
                'title': {
                    'text': 'Distribution of 5 common Categories',
                    'font': dict(
                        family="Courier New, monospace",
                        size=24
                    )
                },
                'paper_bgcolor':'rgb(248, 243, 235)',
                'plot_bgcolor':'rgb(248, 243, 235)',                
                'yaxis': {
                    'title': {
                        'text': "Count",
                        'font': dict(
                            family="Courier New, monospace",
                            size=18,
                            color="#7f7f7f"
                        )
                    }
                },
                'xaxis': {
                    'title': {
                        'text': "Categories",
                        'font': dict(
                            family="Courier New, monospace",
                            size=18,
                            color="#7f7f7f"
                        )
                    }
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()