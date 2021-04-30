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


with header:
    st.title('Content-based Recommender System For Enron Email Dataset')
    st.text("")
    st.text("")
    st.text("")
    st.text('In this project I build a content-based recommender system using the maildir folder')

with dataset:
    st.header('Amalgamation of the maildir folder')
    st.text("")
    st.text('I got this dataset from Kaggle: https://www.kaggle.com/granjithkumar/it-employees-data-for-project-allocation?select=Employee_Designation.csv')

    def column(person):
        import os
        if path.exists("maildir/{}/sent/1.".format(person)):
            file_list = []
            for filename in os.listdir("maildir/{}/sent".format(person)):
                file_list.append(filename)
            file_list = file_list[0:2]
            l = []
            for i in file_list:
                with open('maildir/{}/sent/{}'.format(person, i)) as fo:
                    for rec in fo:
                        l.append(str(rec))
            return l
        else:
            if path.exists("maildir/{}/sent_items/1.".format(person)):
                file_list = []
                for filename in os.listdir("maildir/{}/sent_items".format(person)):
                    file_list.append(filename)
                file_list = file_list[0:2]
                l = []
                for i in file_list:
                    with open('maildir/{}/sent_items/{}'.format(person, i)) as fo:
                        for rec in fo:
                            l.append(str(rec))
                return l
            else: 
                if path.exists("maildir/{}/all_documents/1.".format(person)):
                    file_list = []
                    for filename in os.listdir("maildir/{}/all_documents".format(person)):
                        file_list.append(filename)
                    file_list = file_list[0:2]
                    l = []
                    for i in file_list:
                        with open('maildir/{}/all_documents/{}'.format(person, i)) as fo:
                            for rec in fo:
                                l.append(str(rec))
                    return l
                else:
                    if path.exists("maildir/{}/inbox/1.".format(person)):
                        file_list = []
                        for filename in os.listdir("maildir/{}/inbox".format(person)):
                            file_list.append(filename)
                        file_list = file_list[0:2]
                        l = []
                        for i in file_list:
                            with open('maildir/{}/inbox/{}'.format(person, i)) as fo:
                                for rec in fo:
                                    l.append(str(rec))
                        return l
                    else:
                        if path.exists("maildir/{}/chris_stokley/sent/1.".format(person)):
                            file_list = []
                            for filename in os.listdir("maildir/{}/chris_stokley/sent".format(person)):
                                file_list.append(filename)
                            file_list = file_list[0:2]
                            l = []
                            for i in file_list:
                                with open('maildir/{}/chris_stokley/sent/{}'.format(person, i)) as fo:
                                    for rec in fo:
                                        l.append(str(rec))
                            return l
                        else:
                            print(person)
    list_names = ['allen-p', 'arnold-j', 'arora-h', 'badeer-r', 'bailey-s', 'bass-e', 'baughman-d', 'beck-s', 'benson-r', 'blair-l', 'brawner-s', 'buy-r', 'campbell-l', 'carson-m', 'cash-m', 'causholli-m', 'corman-s', 'crandell-s', 'cuilla-m', 'dasovich-j', 'davis-d', 'dean-c', 'delainey-d', 'derrick-j', 'dickson-s', 'donoho-l', 'donohoe-t', 'dorland-c', 'ermis-f', 'farmer-d', 'fischer-m', 'forney-j', 'fossum-d', 'gang-l', 'gay-r', 'geaccone-t', 'germany-c', 'gilbertsmith-d', 'giron-d', 'griffith-j', 'grigsby-m', 'guzman-m', 'haedicke-m', 'hain-m', 'harris-s', 'hayslett-r', 'heard-m', 'hendrickson-s', 'hernandez-j', 'hodge-j', 'holst-k', 'horton-s', 'hyatt-k', 'hyvl-d', 'jones-t', 'kaminski-v', 'kean-s', 'keavey-p', 'keiser-k', 'king-j', 'kitchen-l', 'kuykendall-t', 'lavorato-j', 'lay-k', 'lenhart-m', 'lewis-a', 'linder-e', 'lokay-m', 'lokey-t', 'love-p', 'lucci-p', 'maggi-m', 'mann-k', 'martin-t', 'may-l', 'mccarty-d', 'mcconnell-m', 'mckay-b', 'mckay-j', 'mclaughlin-e', 'merriss-s', 'meyers-a', 'mims-thurston-p', 'motley-m', 'neal-s', 'nemec-g', 'panus-s', 'parks-j', 'pereira-s', 'perlingiere-d', 'phanis-s', 'pimenov-v', 'platter-p', 'presto-k', 'quenet-j', 'quigley-d', 'rapp-b', 'reitmeyer-j', 'richey-c', 'ring-a', 'ring-r', 'rodrique-r', 'rogers-b', 'ruscitti-k', 'sager-e', 'saibi-e', 'salisbury-h', 'sanchez-m', 'sanders-r', 'scholtes-d', 'schoolcraft-d', 'schwieger-j', 'scott-s', 'semperger-c', 'shackleton-s', 'shankman-j', 'shapiro-r', 'shively-h', 'skilling-j', 'slinger-r', 'smith-m', 'solberg-g', 'south-s', 'staab-t', 'stclair-c', 'steffes-j', 'stepenovitch-j', 'stokley-c', 'storey-g', 'sturm-f', 'swerzbin-m', 'symes-k', 'taylor-m', 'tholt-j', 'thomas-p', 'townsend-j', 'tycholiz-b', 'ward-k', 'watson-k', 'weldon-c', 'whalley-g', 'whalley-l', 'white-s', 'whitt-m', 'williams-j', 'williams-w3', 'wolfe-j', 'ybarbo-p', 'zipper-a', 'zufferli-j']
    data = []
    for i in list_names:
        rey = [i, str(column(i))]
        data.append(rey) 
    Enron_df = pd.DataFrame(data, columns = ['Employee_Name', 'Input']) 
    st.write(Enron_df.head())

    
with features:
    st.header('Features')

    st.markdown('* **Input:** I created this feature in order combine all the relevant parameters required for input')

with model_training:
    st.header('Training')
    st.text('Here you can get recommendation for any individual')

    sel_col, disp_col = st.beta_columns(2)

    input_name = sel_col.selectbox('Select any name from dataset', (Enron_df['Employee_Name']))
    def column(person):
        import os
        if path.exists("maildir/{}/sent/1.".format(person)):
            file_list = []
            for filename in os.listdir("maildir/{}/sent".format(person)):
                file_list.append(filename)
            file_list = file_list[0:2]
            l = []
            for i in file_list:
                with open('maildir/{}/sent/{}'.format(person, i)) as fo:
                    for rec in fo:
                        l.append(str(rec))
            return l
        else:
            if path.exists("maildir/{}/sent_items/1.".format(person)):
                file_list = []
                for filename in os.listdir("maildir/{}/sent_items".format(person)):
                    file_list.append(filename)
                file_list = file_list[0:2]
                l = []
                for i in file_list:
                    with open('maildir/{}/sent_items/{}'.format(person, i)) as fo:
                        for rec in fo:
                            l.append(str(rec))
                return l
            else: 
                if path.exists("maildir/{}/all_documents/1.".format(person)):
                    file_list = []
                    for filename in os.listdir("maildir/{}/all_documents".format(person)):
                        file_list.append(filename)
                    file_list = file_list[0:2]
                    l = []
                    for i in file_list:
                        with open('maildir/{}/all_documents/{}'.format(person, i)) as fo:
                            for rec in fo:
                                l.append(str(rec))
                    return l
                else:
                    if path.exists("maildir/{}/inbox/1.".format(person)):
                        file_list = []
                        for filename in os.listdir("maildir/{}/inbox".format(person)):
                            file_list.append(filename)
                        file_list = file_list[0:2]
                        l = []
                        for i in file_list:
                            with open('maildir/{}/inbox/{}'.format(person, i)) as fo:
                                for rec in fo:
                                    l.append(str(rec))
                        return l
                    else:
                        if path.exists("maildir/{}/chris_stokley/sent/1.".format(person)):
                            file_list = []
                            for filename in os.listdir("maildir/{}/chris_stokley/sent".format(person)):
                                file_list.append(filename)
                            file_list = file_list[0:2]
                            l = []
                            for i in file_list:
                                with open('maildir/{}/chris_stokley/sent/{}'.format(person, i)) as fo:
                                    for rec in fo:
                                        l.append(str(rec))
                            return l
                        else:
                            print(person)
    list_names = ['allen-p', 'arnold-j', 'arora-h', 'badeer-r', 'bailey-s', 'bass-e', 'baughman-d', 'beck-s', 'benson-r', 'blair-l', 'brawner-s', 'buy-r', 'campbell-l', 'carson-m', 'cash-m', 'causholli-m', 'corman-s', 'crandell-s', 'cuilla-m', 'dasovich-j', 'davis-d', 'dean-c', 'delainey-d', 'derrick-j', 'dickson-s', 'donoho-l', 'donohoe-t', 'dorland-c', 'ermis-f', 'farmer-d', 'fischer-m', 'forney-j', 'fossum-d', 'gang-l', 'gay-r', 'geaccone-t', 'germany-c', 'gilbertsmith-d', 'giron-d', 'griffith-j', 'grigsby-m', 'guzman-m', 'haedicke-m', 'hain-m', 'harris-s', 'hayslett-r', 'heard-m', 'hendrickson-s', 'hernandez-j', 'hodge-j', 'holst-k', 'horton-s', 'hyatt-k', 'hyvl-d', 'jones-t', 'kaminski-v', 'kean-s', 'keavey-p', 'keiser-k', 'king-j', 'kitchen-l', 'kuykendall-t', 'lavorato-j', 'lay-k', 'lenhart-m', 'lewis-a', 'linder-e', 'lokay-m', 'lokey-t', 'love-p', 'lucci-p', 'maggi-m', 'mann-k', 'martin-t', 'may-l', 'mccarty-d', 'mcconnell-m', 'mckay-b', 'mckay-j', 'mclaughlin-e', 'merriss-s', 'meyers-a', 'mims-thurston-p', 'motley-m', 'neal-s', 'nemec-g', 'panus-s', 'parks-j', 'pereira-s', 'perlingiere-d', 'phanis-s', 'pimenov-v', 'platter-p', 'presto-k', 'quenet-j', 'quigley-d', 'rapp-b', 'reitmeyer-j', 'richey-c', 'ring-a', 'ring-r', 'rodrique-r', 'rogers-b', 'ruscitti-k', 'sager-e', 'saibi-e', 'salisbury-h', 'sanchez-m', 'sanders-r', 'scholtes-d', 'schoolcraft-d', 'schwieger-j', 'scott-s', 'semperger-c', 'shackleton-s', 'shankman-j', 'shapiro-r', 'shively-h', 'skilling-j', 'slinger-r', 'smith-m', 'solberg-g', 'south-s', 'staab-t', 'stclair-c', 'steffes-j', 'stepenovitch-j', 'stokley-c', 'storey-g', 'sturm-f', 'swerzbin-m', 'symes-k', 'taylor-m', 'tholt-j', 'thomas-p', 'townsend-j', 'tycholiz-b', 'ward-k', 'watson-k', 'weldon-c', 'whalley-g', 'whalley-l', 'white-s', 'whitt-m', 'williams-j', 'williams-w3', 'wolfe-j', 'ybarbo-p', 'zipper-a', 'zufferli-j']
    data = []
    for i in list_names:
        rey = [i, str(column(i))]
        data.append(rey) 
    Enron_df = pd.DataFrame(data, columns = ['Employee_Name', 'Input'])
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

    


