import streamlit as st
from src.projects import drive_cycle_characterisation as dcc
from src.projects import strava_project as sp
from src.projects import public_transport_times as ptt

def run():
    class home_page:
        def __init__(self):
            return None

        def run(self):
            st.title("Projects home")
            st.markdown("### Welcome ! ")
            st.write(
                """This is a collection of short projects that I made as an effort in my Data Science journey."""
            )

    home = home_page()

    projects = {
        "Home": home,
        "Strava": sp,
        "Drive cycle characterisation": dcc,
        "Public transport times": ptt,
    }

    page = st.selectbox(label="Project list", options=list(projects.keys()))

    projects[page].run()
