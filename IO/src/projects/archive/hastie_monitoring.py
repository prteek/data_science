import streamlit as st
from datetime import datetime, timedelta
import time
import boto3
from boto3.dynamodb.conditions import Key
import os
from decimal import Decimal
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import numpy as np
from plotly import graph_objects as go
from sklearn import metrics as mt
from dotenv import load_dotenv
load_dotenv('local_credentials.env')


def dynamodb_to_dict(item):
    """Take json response from a dynamodb query and return a dictionary of appropriate data types"""

    if isinstance(item, dict):
        return {k: dynamodb_to_dict(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [dynamodb_to_dict(l) for l in item]
    elif isinstance(item, Decimal):
        if float(item) % 1 > 0:
            return float(item)
        else:
            return int(item)
    else:
        return item
    
    
def get_data_from_dynamodb(
    key_value,
    table_name,
    primary_key_name="date",
):
    """Convert a json response from dynamodb query to a dictionary"""

    table = boto3.resource(
        "dynamodb",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
    ).Table(table_name)

    response = table.query(KeyConditionExpression=Key(primary_key_name).eq(key_value))
    results_dict = dynamodb_to_dict(response["Items"])

    return results_dict



def download_file_from_s3(bucket_name, file_key, file_name):
    s3 = boto3.resource(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
    )
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(Key=file_key, Filename=file_name)
    return None
    
    
def run():
    st.title('Hastie pipeline monitoring')
    
    st.markdown("""
    Part of Hastie blob classification pipeline project [repo](https://github.com/prteek/ml-pipeline). 
""")
    
    st.sidebar.caption("Hastie AWS Pipeline")
    st.sidebar.image("https://raw.githubusercontent.com/prteek/ml-pipeline/master/metadata/images/pipeline_graph.PNG")
    st.caption("The dashboard monitors performance of model that the pipeline re-trains everyday (on AWS Sagemaker) and uses to make predictions every 5 minutes")
    
    # Get model and training data
    download_file_from_s3('hastie', 'preprocess/data/train.parquet', 'train.parquet')
    download_file_from_s3('hastie', 'preprocess/data/test.parquet', 'test.parquet')
    
    df_train = pd.read_parquet('train.parquet')
    df_test = pd.read_parquet('test.parquet')
    
    download_file_from_s3('hastie', 'model/model.mdl', 'model.mdl')
    model = joblib.load('model.mdl')
    
    predictors = ['x1', 'x2']
    target = 'is_blue'
    X = df_train[predictors].to_numpy()
    y = df_train[target].values
    
    
    # Recent predictions
    table_name = 'hastie'
    date = datetime.now().strftime('%Y-%m-%d')
    data = get_data_from_dynamodb(date, table_name)
    
    if len(data):
        df = pd.DataFrame(data)[-30:]
        df['time'] = pd.to_datetime(df['time'])        
        
        labels = ["Correctly predicted", "Incorrect"]
        correct_prediction_label = {True: labels[0], False: labels[1]}
        df['is_correct'] = [correct_prediction_label[i] for i in (df['prediction'] == df['is_blue'])]
        fig = make_subplots(rows=1,cols=2,
                            subplot_titles=["Recent Predictions", 
                                            "Accuracy of predictions"],
                            specs=[[{"type": "xy"}, {"type": "domain"}]],)
            
        fig.add_traces(px.scatter(df, x='time', y='prediction', color='is_correct', 
                         color_discrete_map={labels[0]:'#636EFA', 
                                             labels[1]:'#EF553B'}).data)
        
        percent_correct = (df['prediction'] == df['is_blue']).sum()/len(df)*100
        fig.add_trace(
            go.Pie(
                values=np.round([percent_correct, 100-percent_correct], 1),
                labels=labels,
            ),
            row=1,
            col=2,
        )
        
        st.plotly_chart(fig)
        
        col1, col2, col3 = st.columns(3)
        accuracy_train = mt.accuracy_score(y, model.predict(X))
        accuracy_predict = mt.accuracy_score(df['is_blue'], df['prediction'])
        col1.metric("Accuracy of prediction", f"{round(accuracy_predict,2)}", delta=f"{round(accuracy_predict - accuracy_train,2)}: Delta from train")
        
        precision_train = mt.precision_score(y, model.predict(X))
        precision_predict = mt.precision_score(df['is_blue'], df['prediction'])
        col2.metric("Precision of prediction", f"{round(precision_predict,2)}", delta=f"{round(precision_predict - precision_train,2)}: Delta from train")
        
        recall_train = mt.recall_score(y, model.predict(X))
        recall_predict = mt.recall_score(df['is_blue'], df['prediction'])
        col3.metric("Recall of prediction", f"{round(recall_predict,2)}", delta=f"{round(recall_predict - recall_train,2)}: Delta from train")
        
        st.markdown("---")
    else:
        st.markdown("### No predictions made today")
        
    # Decision boundary learnt    
    # Create a mesh grid on which we will run our model
    mesh_size = .1
    margin = 0
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)

    # Create classifier, run predictions on grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    df_grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel(), Z], columns = predictors+[target])
        
    # Plot the figure
    colormap = {0:'#EF553B', 1:'#636EFA'}
    fig = go.Figure() 
    for i in [0,1]:
        grid_i = df_grid.query(f'is_blue=={i}')
        train_i = df_train.query(f'is_blue=={i}')
        test_i = df_test.query(f'is_blue=={i}')
        
        fig.add_scatter(x=grid_i['x1'], y=grid_i['x2'], mode='markers',
                       marker={'color':colormap[i], 'symbol':'square'}, 
                        opacity=0.5, 
                        showlegend=False)
        fig.add_scatter(x=train_i['x1'], y=train_i['x2'], mode='markers',
                       marker={'color':colormap[i], 'symbol':'circle', 'size':8}, 
                        opacity=0.7, name='Train')
        fig.add_scatter(x=test_i['x1'], y=test_i['x2'], mode='markers',
                       marker={'color':colormap[i], 'symbol':'diamond', 'size':8}, 
                        opacity=1, name='Test')
   
    
    fig.update_layout({'title':f'Decision boundary learnt on {date}',
                      'xaxis_title':'x1',
                      'yaxis_title':'x2'})
    st.plotly_chart(fig)
    
    
    if len(data):
        fig = go.Figure()
    
        fig.add_trace(go.Histogram(x=df_train['is_blue'], name='Training data', histnorm='percent'))
        fig.add_trace(go.Histogram(x=df['prediction'], name='Predictions made', histnorm='percent'))
    
        fig.update_layout({'title':'Distribution of Target variable',
                      'xaxis_title':'target variable value',
                      'yaxis_title':'% of total instances'})
        st.plotly_chart(fig)
    else:
        pass
    
    
    
    st.markdown(""" ###
    ---
    [Prateek](https://www.linkedin.com/in/prteek/ "LinkedIn")  
    [Repository](https://github.com/prteek/IO/ "Github")

    """)