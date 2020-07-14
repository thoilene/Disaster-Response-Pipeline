import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar,Heatmap
from plotly.graph_objs.layout import Annotation 
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def disaster_counts(df_):
    Y_ = df_.drop(['id','message','original','genre'],axis=1)
    disaster_0 = []
    disaster_1 = []
    disaster_2 = []
    disaster_1_2 = []
    for col in Y_.columns:
        disaster_0.append(Y_[Y_[col]==0].shape[0])
        disaster_1.append(Y_[Y_[col]==1].shape[0])
        disaster_2.append(Y_[Y_[col]==2].shape[0])
        disaster_1_2.append(Y_[Y_[col]==1].shape[0]+Y_[Y_[col]==2].shape[0])
    
    return pd.DataFrame(index=Y_.columns, data={'disasters_type_0':disaster_0,'disasters_type_1':disaster_1,'disasters_type_2':disaster_2,'disasters_type_1_2':disaster_1_2})



# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponseMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    d_counts = disaster_counts(df)
    
    categories = list(d_counts.index)
    disasters_type_1 = list(d_counts['disasters_type_1'].values)
    disasters_type_1_2 = list(d_counts['disasters_type_1_2'].values)
    
    
    z = df[['related','request','aid_related','weather_related','direct_report']].corr()
    z = z.values
    x=['related','request','aid_related','weather_related','direct_report']
    y=['related','request','aid_related','weather_related','direct_report']
    z_vals = z
    
    annotations = []
    for n, row in enumerate(z):
        #print(n, row)
        for m, val in enumerate(row):
            #print(n,m,z[n][m],x[m],y[n])
            annotations.append(Annotation(text="{:.2f}".format(z[n][m]), x=x[m], y=y[n],
                                             xref='x1', yref='y1', showarrow=False))
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # second element
        {
            'data': [
                Bar(
                    x=categories,
                    y=disasters_type_1,
                    name="type 1 only"
                ),
                Bar(
                    x=categories,
                    y=disasters_type_1_2,
                    name="type 1/2"
                )
            ],

            'layout': {
                'title': 'Disaster Count (types 1 and 2)',
                'yaxis': {
                    'title': "Disaster Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        # third element
        {
            'data': [
                Heatmap(
                   z = z_vals,
                   x=['related','request','aid_related','weather_related','direct_report'],
                   y=['related','request','aid_related','weather_related','direct_report'],
                   colorscale=[[0.0, '#F5FFFA'],[0.2, '#ADD8E6'], [0.4, '#87CEEB'],[0.6, '#87CEFA'], [0.8, '#40E0D0'], [1.0, '#00CED1']],
                   hoverongaps = False)
            ],

            'layout': {
                'title': 'Correlations between the most important categories',
                'annotations':annotations,
                
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
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()