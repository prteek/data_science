import streamlit as st
import boto3
import numpy as np
from plotly import graph_objects as go
import awswrangler as wr
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer


boto3_session = boto3.Session(region_name="eu-west-1")
PREDICTORS = [
    "moving_time",
    "average_heartrate",
]  # Check with ML model implementation or create a dependency

DB = 'strava'
SUFFER_SCORE_TABLE = 'predicted_suffer_score'
FITNESS_TABLE = 'predicted_fitness_score'

predictor_suffer_score = Predictor("strava", serializer=CSVSerializer())
predictor_fitness = Predictor("strava-fitness", serializer=CSVSerializer())


@st.cache_data
def fetch_fitness_data(date_end="2021-04-01"):
    data = wr.athena.read_sql_query(f"""
                    SELECT * FROM strava.predicted_fitness_score WHERE start_timestamp <= date('{date_end}')""",
                                    database=DB,
                                    boto3_session=boto3_session).sort_values("start_timestamp")

    return data


def run():
    st.title('Fitness simulation')
    st.markdown("""
    Part of Strava metrics modelling project [repo](https://github.com/prteek/strava-project).
""")

    # Get fitness data
    date_today = np.datetime_as_string(np.datetime64("today") + np.timedelta64(1, 'D'))  # Add 1 day to today's date to include today's workout
    df_fitness = fetch_fitness_data(date_today)
    fitness_scores = df_fitness[["fitness_score_pre", "fitness_score"]].values.ravel()
    start_dates = df_fitness[["start_timestamp", "start_timestamp"]].values.ravel()
    fig = go.Figure()
    fig.add_scatter(x=start_dates,
                    y=fitness_scores,
                    mode='lines+markers',
                    name="Fitness score",
                    marker=dict(color="#636EFA"),
                    line=dict(color='gray', dash="dot"))

    # Add vertical line on plot indicating today's date
    fig.add_scatter(x=[np.datetime64("now"), np.datetime64("now")],
                    y=[0,15], mode="lines", line=dict(color="yellow", width=1, dash="dot"), name="Today")

    ini = df_fitness['fitness_score'].values[-1]
    last_date = df_fitness['start_timestamp'].values[-1]
    days_since_last_workout = np.timedelta64(np.datetime64("now") - last_date, 'D').astype(int)

    st.caption("Workout 1")
    col1, col2, col3, col4 = st.columns(4)
    days_from_today_1 = int(col1.text_input("Days from today", value=2, key=11))
    mov_time = col2.text_input("Moving time (minutes)", value=0, key=12)
    avg_hr = col3.text_input("Average heart rate", value=0, key=13)
    payload_ss = [[float(mov_time)*60, float(avg_hr)]]
    suffer_score_1 = eval(predictor_suffer_score.predict(payload_ss).decode())[0]
    payload_fitness = [[ini, int(days_from_today_1 + days_since_last_workout), suffer_score_1]] # adjusting for total days since last activity
    fitness_score_1 = eval(predictor_fitness.predict(payload_fitness).decode())[0]
    col4.metric("Predicted suffer score", round(suffer_score_1, 0))

    st.caption("Workout 2")
    col1, col2, col3, col4 = st.columns(4)
    days_from_today_2 = int(col1.text_input("Days from today", value=days_from_today_1+2, key=21))
    mov_time = col2.text_input("Moving time (minutes)", value=0, key=22)
    avg_hr = col3.text_input("Average heart rate", value=0, key=23)
    payload_ss = [[float(mov_time)*60, float(avg_hr)]]
    suffer_score_2 = eval(predictor_suffer_score.predict(payload_ss).decode())[0]
    payload_fitness = [[fitness_score_1[1], int(days_from_today_2) - int(days_from_today_1), suffer_score_2]]
    fitness_score_2 = eval(predictor_fitness.predict(payload_fitness).decode())[0]
    col4.metric("Predicted suffer score", round(suffer_score_2, 0))

    st.caption("Workout 3")
    col1, col2, col3, col4 = st.columns(4)
    days_from_today_3 = int(col1.text_input("Days from today", value=days_from_today_2+2, key=31))
    mov_time = col2.text_input("Moving time (minutes)", value=0, key=32)
    avg_hr = col3.text_input("Average heart rate", value=0, key=33)
    payload_ss = [[float(mov_time)*60, float(avg_hr)]]
    suffer_score_3 = eval(predictor_suffer_score.predict(payload_ss).decode())[0]
    payload_fitness = [[fitness_score_2[1], int(days_from_today_3) - int(days_from_today_2), suffer_score_3]]
    fitness_score_3 = eval(predictor_fitness.predict(payload_fitness).decode())[0]
    col4.metric("Predicted suffer score", round(suffer_score_3, 0))

    extend_in_future = [days_from_today_1, days_from_today_2, days_from_today_3]
    future_dates_ = np.array([np.datetime64("now") + np.timedelta64(i, 'D') for i in extend_in_future])
    future_dates = np.c_[future_dates_,future_dates_].ravel()
    future_fitness = np.round(np.array([fitness_score_1, fitness_score_2, fitness_score_3]).ravel(),2)

    fig.add_scatter(x=future_dates,
                    y=future_fitness,
                    mode='lines + markers',
                    name="Predicted fitness score",
                    line=dict(color='#FFA15A'))

    fig.update_layout(xaxis_title="Date",
                      yaxis_title="Fitness score",
                      title="Fitness scores over time")

    st.plotly_chart(fig)

    st.markdown("---")










