# author           : Prateek
# email            : prateekpatel.in@gmail.com
# description      : Just for fun determination of a polynomial based on input name

import string as strng
import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction
import streamlit as st

def run():   
    st.title('Your personal Polynomial')

    name = st.text_input("Reveal thy name and I shall reveal thy polynomial", "Gandalf")


    # extract only alphabets from name (just in case)
    name_lower = [char for char in name.lower() if char.isalpha()]
    degree_of_poly = len(name_lower) - 1

    alphabets = list(strng.ascii_lowercase)

    y = [
        i + 1
        for letter in name_lower
        for i, alphabet in enumerate(alphabets)
        if letter == alphabet
    ]
    x = [i + 1 for i in range(degree_of_poly + 1)]

    z = np.polyfit(x, y, degree_of_poly)
    coefficients = [str(round(Fraction(i), 3)) for i in z]

    f = np.poly1d(z)
    # calculate new x's and y's
    x_new = np.linspace(x[0], x[-1], 50)
    y_new = f(x_new)


    plt.scatter(x, y)
    # label each point
    for letter, x_i, y_i in zip(name_lower, x, y):
        plt.annotate(
            letter,
            xy=(x_i, y_i),  # Put the label with its point
            xytext=(10, -10),  # but slightly offset
            textcoords="offset points",
        )
    plt.figure()
    plt.plot(x_new, y_new)
    plt.title("Your personal polynomial")
    plt.grid()
    plt.show()

    st.pyplot()

    # In[4]:


    string = ""
    for i, coeff in enumerate(coefficients):
        if eval(coeff) >= 0:
            string = string + coeff + "  x^" + str(degree_of_poly - i) + "  +  "
        else:
            string = string[:-3] + "  -  " + coeff[1:] + "  x^" + str(degree_of_poly - i) + "  +  "

    string = string[:-3]

    text = "Your personal polynomial: " + string
    st.markdown(f""" Your personal polynomial:  
    #### {string}
    """)

    print(string)

    
    st.markdown(""" ###
    ---
    [Prateek](https://www.linkedin.com/in/prteek/ "LinkedIn")  
    [Repository](https://github.com/prteek/IO/ "Github")

    """)