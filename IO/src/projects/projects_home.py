import streamlit as st
from src.projects import drive_cycle_characterisation as dcc
from src.projects import strava_project as sp
from src.projects import public_transport_range as ptr
from src.projects import euro_2024_modelling as e2m

def run():
    class home_page:
        def __init__(self):
            return None

        def run(self):
            st.title("Projects home")
            st.markdown("### Welcome ! ")
            st.write(
                """This is a collection of short projects that I've made in my own time.  
There is a slim chance you may also enjoy these.                 
"""
            )

    home = home_page()

    projects = {
        "Home": home,
        'Euro 2024 model': e2m,
        "Strava": sp,
        "Drive cycle characterisation": dcc,
        "Public transport range": ptr,
    }

    page = st.selectbox(label="Project list", options=list(projects.keys()))

    projects[page].run()
