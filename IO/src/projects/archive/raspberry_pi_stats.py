# author           : Prateek
# email            : prateekpatel.in@gmail.com
# description      : Just for fun determination of a polynomial based on input name

import streamlit as st
import webbrowser

def run():
    st.title('Raspberry pi dashboard')
    st.markdown(""" ####
    *Dashboard hosted at* :  [https://prateek-rpi4-stats.anvil.app](https://prateek-rpi4-stats.anvil.app)
    """)
    
    url = "https://prateek-rpi4-stats.anvil.app"
    webbrowser.open(url)