import streamlit as st
import boto3
import os
from plotly.subplots import make_subplots
import numpy as np
from plotly import graph_objects as go
from sklearn import metrics as mt
import awswrangler as wr
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer

expected_error = lambda y_true, y_predicted: np.mean(y_predicted - y_true)
boto3_session = boto3.Session(region_name="eu-west-1")
bucket = "pp-strava-data"
TARGET = "suffer_score"
PREDICTORS = [
    "moving_time",
    "average_heartrate",
]  # Check with ML model implementation or create a dependency
PREDICTED = "predicted_suffer_score"

DB = "strava"
TABLE = "predicted_suffer_score"

sm_client = boto3.client("sagemaker")
predictor = Predictor("strava", serializer=CSVSerializer())
model_name = predictor._get_model_names()[0]
train_date = sm_client.describe_model(ModelName=model_name)["CreationTime"].strftime(
    "%Y-%m-%d"
)
train_date = "2023-04-01"


@st.cache_data
def fetch_training_data():
    df = wr.s3.read_csv(
        os.path.join("s3://", bucket, "prepare-training-data", "output", "train.csv"),
        boto3_session=boto3_session,
    )
    return df


@st.cache_data
def fetch_recent_predictions(train_date):
    df = wr.athena.read_sql_query(
        f"""
                    SELECT * FROM strava.predicted_suffer_score
                    JOIN strava.activities
                    ON strava.predicted_suffer_score.activity_id = strava.activities.id
                    WHERE start_timestamp >= date('{train_date}')
                    """,
        database=DB,
        boto3_session=boto3_session,
    )
    return df


def run():
    st.title("Suffer score modelling")
    st.markdown(
        """
    Part of Strava metrics modelling project [repo](https://github.com/prteek/strava-project).
"""
    )

    # st.caption("Strava pipeline")
    # st.image("https://raw.githubusercontent.com/prteek/strava-project/main/resources/strava.png")
    st.caption(
        """The dashboard monitors performance of model that the pipeline re-trains week (on AWS Sagemaker) and uses to make predictions for each workout.
    This project started out as my curiosity to model my health data. After looking at fascinating stats from Strava app,
    I decided to model ```Suffer Score``` assigned to each workout by Strava. This being a paid feature could mean that if I
    model it well, I may actually end up saving money and so far it's going well.
    """
    )

    # Get model and training data
    df_train = fetch_training_data()

    X = df_train[PREDICTORS].values
    y = df_train[TARGET].values
    y_pred = np.array(eval(predictor.predict(X).decode()), dtype=float)

    # Recent predictions
    data = fetch_recent_predictions(train_date)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Predicted vs actual suffer score",
            "Distribution of errors (unseen data)",
        ),
    )

    fig.add_scatter(
        x=y,
        y=y_pred,
        mode="markers",
        name="Training data",
        marker=dict(size=10, opacity=0.5),
    )

    fig.add_scatter(
        x=data[TARGET],
        y=data[PREDICTED],
        mode="markers",
        name="Unseen data",
        marker=dict(color="yellow", size=10, opacity=0.8),
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            name="Perfect prediction",
            line=dict(color="green", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        {
            "xaxis": {"title": "Actual suffer score", "range": [0, 100]},
            "yaxis": {"title": "Predicted suffer score", "range": [0, 100]},
        }
    )

    fig.add_trace(
        go.Histogram(
            x=data[PREDICTED] - data[TARGET],
            name="Error",
            xbins=dict(start=-20, end=20, size=5),
            marker=dict(color="azure", opacity=0.5),
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        {
            "xaxis2": {"title": "Error", "range": [-20, 20]},
        }
    )

    st.plotly_chart(fig)

    col1, col2, col3 = st.columns(3)

    mae_train = mt.mean_absolute_error(y, y_pred)
    mae_test = mt.mean_absolute_error(data[TARGET], data[PREDICTED])
    col1.metric(
        "MAE of prediction",
        f"{round(mae_test,1)}",
        delta=f"{round(mae_test - mae_train,1)}: Delta from train",
        delta_color="inverse",
    )

    rmse_train = np.sqrt(mt.mean_squared_error(y, y_pred))
    rmse_test = np.sqrt(mt.mean_squared_error(data[TARGET], data[PREDICTED]))
    col2.metric(
        "RMS Error of prediction",
        f"{round(rmse_test,1)}",
        delta=f"{round(rmse_test - rmse_train,1)}: Delta from train",
        delta_color="inverse",
    )

    r2_train = mt.r2_score(y, y_pred)
    r2_test = mt.r2_score(data[TARGET], data[PREDICTED])
    col3.metric(
        "R2 score of prediction",
        f"{round(r2_test,2)}",
        delta=f"{round(r2_test - r2_train,2)}: Delta from train",
    )

    st.markdown("---")

    st.subheader("Suffer score relationship")
    fig = go.Figure()
    fig.add_scatter(
        x=np.round(X[:, 0] / 60, 1),
        y=X[:, 1],
        text=np.round(y_pred, 0),
        textfont=dict(color="yellow"),
        mode="markers+text",
        textposition="bottom center",
        name="Training data",
    )
    fig.update_layout(
        {
            "xaxis": {"title": "Moving time (minutes)"},
            "yaxis": {"title": "Average heart rate"},
        },
        title="Effect of heart rate and moving time on suffer score",
    )

    mov_time_vec = np.arange(0, 65, 5) * 60
    avg_hr_vec = np.arange(50, 200, 20)
    xv, yv = np.meshgrid(mov_time_vec, avg_hr_vec)
    X_ = np.c_[xv.ravel(), yv.ravel()]
    y_mesh = np.array(eval(predictor.predict(X_).decode()), dtype=float).reshape(
        xv.shape
    )
    fig.add_trace(
        go.Contour(
            x=np.round(mov_time_vec / 60, 1),
            y=avg_hr_vec,
            z=y_mesh,
            colorscale="Hot",
            contours=dict(
                start=0,
                end=115,
                size=5,
            ),
        )
    )

    st.plotly_chart(fig)

    st.markdown("---")

    st.subheader("Suffer score calculator")
    mov_time = st.text_input("Moving time (minutes)", value=20)
    avg_hr = st.text_input("Average heart rate", value=150)
    go_calc = st.button("Calculate")

    if go_calc:
        X_manual = [[int(mov_time) * 60, int(avg_hr)]]
        y_pred = eval(predictor.predict(X_manual).decode())[0]
        st.metric("Suffer score", f"{round(y_pred,1)}")

    st.markdown("---")
