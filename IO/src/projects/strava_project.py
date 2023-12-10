import streamlit as st
from src.projects import strava_suffer_score as sss
from src.projects import strava_fitness as sf


def run():
    fitness_simulation_tab, suffer_score_model_tab = st.tabs(["Fitness simulation", "Suffer score modelling"])

    with fitness_simulation_tab:
        sf.run()

    with suffer_score_model_tab:
        sss.run()
