import streamlit as st
import pandas as pd
import os
import os.path
from os import path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances


header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

@st.cache
def get_data(filename):
    HR_DATA = pd.read_csv(filename)
    return HR_DATA

with header:
    st.title('Content-based Recommender System For Enron Email Dataset')
    st.text("")
    st.text("")
    st.text("")
    st.text('In this project I build a content-based recommender system using the maildir folder')

with dataset:
    st.header('Amalgamation of the maildir folder')
    st.text("")
    st.text('I got this dataset from this GitHub link: http://www.cs.cmu.edu/~./enron/')

    Enron_df = pd.read_csv ('data/enron_email_dataset.csv')
    st.write(Enron_df.head())

with features:
    st.header('Features')

    st.markdown('* **Input:** I created this feature in order combine all the relevant parameters required for input')

with model_training:
    st.header('Training')
    st.text('Here you can get recommendation for any individual')

    sel_col, disp_col = st.beta_columns(2)

    input_name = sel_col.selectbox('Select any name from dataset', (Enron_df['Employee_Name']))
     
    Enron_df = pd.read_csv ('data/enron_email_dataset.csv')
    sel_col.subheader('Name Details')
    sel_col.write(Enron_df.loc[Enron_df['Employee_Name'] == input_name])

    
    metadata = Enron_df.copy()
    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')
    #Replace NaN with an empty string
    metadata['Input'] = metadata['Input'].fillna('')
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(metadata['Input'])
    cosine_similarity(tfidf_matrix)
    cosine_distances(tfidf_matrix)
    cosine_model=cosine_similarity(tfidf_matrix)
    cosine_model_df=pd.DataFrame(cosine_model,index=Enron_df.Employee_Name,columns=Enron_df.Employee_Name)
    cosine_model_df.head()
    def make_recommendations(movie_user_likes):
        return cosine_model_df[movie_user_likes].sort_values(ascending=False)[:10]


    recommender = make_recommendations(input_name)

    disp_col.subheader('Recommendations')
    recommender = recommender.to_frame()
    recommender.reset_index(level=0, inplace=True)
    recommender = recommender.rename(columns={input_name: "Cosine Similarity"})
    disp_col.write(recommender)

    disp_col.subheader('Recommendations Details')
    for i in recommender.Employee_Name:
        disp_col.write(Enron_df.loc[Enron_df['Employee_Name'] == i])

    


