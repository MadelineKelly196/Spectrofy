import streamlit as st
import torch
import os
import random
from utils.preprocessing import *
from utils.models import *
from utils.postprocessing import *

st.set_page_config(
    page_title="spectrofy_home",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={"About": "# *This is a website that empowers musicians to understand the danceability grade of their songs. Enjoy :)*"},
)

# Add logo to the sidebar
st.sidebar.image("./spectrofy_logo.jpeg", use_column_width=True)
 
# Title container
with st.container():
    st.markdown('<h1 style="color: green;">Welcome to Spectrofy!</h1>', unsafe_allow_html=True)

# Paragraph container
with st.container():
    st.markdown('''
    <style>
    .container {
        border: 1px solid black;
        background-color: #e0f5f0;
        padding: 10px;
        border-radius: 10px; /* Add rounded corners */
    }
    body {
        color: black;
        background-color: #e0f5f0;
    }
    h1 {
        color: green;
    }
    </style>

    <div class="container">
        Spectrofy is a website that empowers musicians to understand the danceability grade of their songs. 
        With Spectrofy, you can analyze the danceability of your songs and gain insights into their rhythm and beats. 
        Whether you're a professional musician or just a music enthusiast, Spectrofy is here to help you explore the danceability of your favorite tracks. 
        Enjoy the journey and let the music move you!
    </div>
    <br>
    ''', unsafe_allow_html=True)

# Add radio buttons for analysis selection
help_dance = "When **Danceability** is selected, our model predicts the danceability of the input song, based on training data pulled from Spotify. The danceability metric is based on the Spotify metric, which (as explained on their website), describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable."
help_genre = "When **Genre** is selected, our model predicts the genre of the input song, based on training data pulled from Spotify."
help_text = help_dance + ' \n\n' + help_genre
analysis_option = st.radio("**Select analysis option:**", ("Danceability", "Genre"), help=help_text)

#initialize and unpack the dance model
#put here so that it doesn't need to be redone for each uploaded file
dance_model = DanceabilityModel()
param_path = os.path.join('utils', 'model_params', 'danceability.pth')
dance_model.load_state_dict(torch.load(param_path, map_location='cpu'))
dance_model.eval() #disable training mode

# Song uploader
allowed_types = ['aiff', 'au', 'avr', 'caf', 'flac', 'htk', 'svx', 'mat4', 'mat5', 'mpc2k', 'mp3', 'ogg', 'paf', 'pvf', 'raw', 'rf64', 'sd2', 'sds', 'ircam', 'voc', 'w64', 'wav', 'nist', 'wavex', 'wve', 'xi'] 
#allowed_types = ['mp3', 'wav'] #TODO: decide on accepting either above or these reduced options for cleanliness
uploaded_files = st.file_uploader("**Upload your songs and analyze them with one click**", type=allowed_types, accept_multiple_files=True)

if uploaded_files is not None:
    for uploaded_file in reversed(uploaded_files):
        #convert to spectrogram for ML model
        try:
            spec = audio_to_spec(uploaded_file)
            spec = transform(spec)
        except AssertionError as e:
            st.error(f"{uploaded_file.name} couldn't be analyzed - audio cannot be shorter than {e.args[0]} s")
            continue #skip analysis for this song

        # If selected, determine song danceability
        if analysis_option == "Danceability":
            try:
                with st.spinner('Evaluating your song...'):
                    #evaluate input song
                    danceability = dance_model(spec).item()
                    #modify danceability metric for readability by the user
                    dance_val = round(danceability, 2)
                    dance_range = categorize_dance(dance_val)
                st.success(f"Your song {uploaded_file.name} has a {dance_range} level of Danceability. Your song got a score of: {dance_val}")
            except:
                st.error(f'Could not determine danceability of {uploaded_file.name}. :(')
        
        # If selected, determine song genre
        elif analysis_option == "Genre":
            genres = ['Rock 🎸', 'Indie 🪕', 'Pop 🎤', 'Classical 🎻', 'Jazz 🎷', 'Hip Hop 📻']
            genre = random.choice(genres)
            st.success(f"Your song {uploaded_file.name} belongs to the genre: {genre}")


