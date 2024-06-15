import pandas as pd
import numpy as np
import json
from joblib import load
import requests
import shap
from streamlit_shap import st_shap
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Titres et logo
def render_title() -> None:
    st.set_page_config(page_title="Dashboard : scoring des candidats à un prêt", page_icon=":bar_chart:",layout="wide")

    col1, col2 = st.columns((9,1))
    with col1:
        st.title(":bar_chart: Dashboard")
        st.markdown(
            "<h5 style='font-weight: normal; font-size: 24px;'>Ce tableau de bord (Production) permet d'afficher le scoring en % de chaque candidat au prêt.Le scoring s'appuie sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.). <br>Ce dashboard interactif permet aux chargés de relation client de pouvoir expliquer de façon la plus transparente possible les décisions d’octroi de crédit. Il affiche donc les criteres qui ont conduit à ce resultat.</h5><br>", 
            unsafe_allow_html=True
        )
    with col2:
        st.image('Logo_projet7.jpg', width=150)


# Chargement des données
def load_data(base_dir):
    data_to_be_loaded = os.path.join(base_dir, 'X_test50_withIndex.csv')
    data = pd.read_csv(data_to_be_loaded) # mettre juste le nom du fichier

    # Convertir les ID clients en entiers
    data['SK_ID_CURR'] = data['SK_ID_CURR'].astype(int)
    return data

# Chargement du glossaire des principales features
def load_glossaire(base_dir):
    glossaire_to_be_loaded = os.path.join(base_dir, 'Glossaire.xlsx')
    glossaire = pd.read_excel(glossaire_to_be_loaded, index_col=None)
    return glossaire

# Selection du candidat dans liste deroulante et formattage attendu par API
def select_candidat():
    st.sidebar.header("Sélection :")
    candidat = st.sidebar.selectbox("Choisir le candidat au prêt :", data['SK_ID_CURR'].unique())
    df = data.loc[data['SK_ID_CURR']==candidat]
    df = df.drop(columns=['SK_ID_CURR'])
    df_dict = df.to_dict(orient='records')[0] #transform en format json
    return df_dict

# Appel a l'API pour obtenir la prédiction
def prediction(row):
    prediction_url = 'https://myfirstapi-cd3-2f92f3a57712.herokuapp.com/prediction' # a remplacer avec nouvelle adresse Heroku
    # Envoyer une requête POST à l'API avec les données JSON dans le corps de la requête
    response = requests.post(prediction_url, json=row)
    response_data = json.loads(response.text)
    prediction_values = response_data["prediction"]
    return prediction_values

# Affichage du resultat de la prediction
def affich_predict(response):
    prediction = response[1]
    if prediction == 1:
        st.markdown(
            "<h3 style='font-weight: bold; font-size: 24px; color: red;'>Demande de crédit refusée</h3>", 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h3 style='font-weight: bold; font-size: 24px; color: green;'>Demande de crédit accordée</h3>", 
            unsafe_allow_html=True
        )

# Affichage de la probabilite de remboursement
def affich_predict_proba(response):
    prediction = np.round(response[0][0]*100, 2)
    st.metric(label="Probabilité de remboursement :", value=f"{prediction} %")

# Affichage d'un pie chart
def pie_chart(response):
    labels = 'Remboursement', 'Non-remboursement'
    sizes = [response[0][0],response[0][1]]

    fig1, ax1 = plt.subplots(figsize=(10, 10))
    pie_wedge_collection, text_props, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                   startangle=90, textprops={'fontsize': 20}) 
    ax1.axis('equal')  # Assure un rapport d'aspect égal pour dessiner un cercle.
    ax1.set_title('Répartition des prédictions', fontsize=22)  
    legend = ax1.legend(fontsize=20, bbox_to_anchor=(1, 0.5))  
    legend.set_title('Légende', prop={'size': 20}) 

    # Ajuste la taille des labels
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(22)

    st.pyplot(fig1)

def box_plot(top_features,selected_index):
    # Créer un sous-ensemble de données avec les top_features
    sub_data = data_woIndex[top_features]

    # Normaliser les données pour les mettre sur une échelle comparable
    scaled_data = (sub_data - sub_data.mean()) / sub_data.std()

    # Tracer le box plot avec des échelles différentes pour chaque axe y
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=scaled_data, ax=plt.gca(), orient='h')
    ax.set_ylabel('Top 5 variables')
    ax.set_xlabel('Valeurs normalisées')
    plt.title('Box Plot des top 5 features')
    # Ajouter les marqueurs pour les valeurs du candidat sélectionné
    for i, feature in enumerate(top_features):
        ax.plot(scaled_data.iloc[selected_index, i], i, 'ro')  # Marqueur rouge pour la valeur du candidat sélectionné

    st.pyplot(plt)  # Afficher le graphique dans Streamlit


if __name__ == "__main__":
    render_title()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    data = load_data(base_dir)
    glossaire = load_glossaire(base_dir)
    data_woIndex = data.drop(columns=['SK_ID_CURR'])
    row_to_send = select_candidat()
    response = prediction(row_to_send)
    affich_predict(response)

    # Obtenir l'index du candidat sélectionné
    selected_index = data_woIndex[data_woIndex.eq(pd.Series(row_to_send)).all(axis=1)].index[0]

    shap_values_to_be_loaded = os.path.join(base_dir, 'shap_values.joblib')
    shap_values = load(shap_values_to_be_loaded)
    shap_data_to_be_loaded = os.path.join(base_dir, 'shap_data.joblib')
    shap_data = load(shap_data_to_be_loaded)

    top_features = data_woIndex.columns[np.argsort(np.abs(shap_values.values).mean(0))][::-1][:5]

    col1, col2 = st.columns((1,8))
    with col1:
        affich_predict_proba(response)
    with col2:
        st.markdown(
            "<h4 style='font-weight: normal; font-size: 24px;'>Force Plot :</h4>", 
            unsafe_allow_html=True
        )
        st.markdown(
        "<p style='font-weight: normal; font-size: 16px;'>Le Force Plot est un outil de visualisation qui permet de comprendre l'impact de chaque feature sur la prédiction du modèle. Chaque barre horizontale représente la contribution de la feature correspondante à la prédiction. Les features à droite diminuent la prédiction, tandis que celles à gauche l'augmentent.</p>",
        unsafe_allow_html=True
        )
        st_shap(shap.force_plot(shap_data['expected_value'], shap_data['shap_values'][selected_index,:], data_woIndex.iloc[selected_index,:]), height=200, width=1800) 
    
    col1, col2 = st.columns((2.8,2))
    with col1:
        #st.markdown("<br>", unsafe_allow_html=True)
        #st_shap(shap.summary_plot(shap_values, data_woIndex, plot_type="bar"), height=450)
        st.markdown(
            "<h4 style='font-weight: normal; font-size: 24px;'>Waterfall Plot :</h4>", 
            unsafe_allow_html=True
        )
        st.markdown(
        "<p style='font-weight: normal; font-size: 16px;'>Le Waterfall Plot est une autre forme de visualisation du Force Plot ci-dessus.</p>",
        unsafe_allow_html=True
        )
        st_shap(shap.plots.waterfall(shap_values[selected_index]), height=400, width=1200)
    with col2:
        st.markdown(
            "<h4 style='font-weight: normal; font-size: 24px;'>Decision Plot :</h4>", 
            unsafe_allow_html=True
        )
        st.markdown(
        "<p style='font-weight: normal; font-size: 16px;'>Le decision plot illustre graphiquement l'importance des différentes caractéristiques dans la prise de décision du modèle.</p>",
        unsafe_allow_html=True
        )
        st_shap(shap.decision_plot(shap_data['expected_value'], shap_data['shap_values'][selected_index,:], feature_names=data_woIndex.columns.tolist()), height=600, width=900)

    col1, col2 = st.columns((2.5,2))
    with col1:
        st.markdown(
            "<h4 style='font-weight: normal; font-size: 24px;'>Graphique des barres SHAP :</h4>", 
            unsafe_allow_html=True
        )
        st.markdown(
        "<p style='font-weight: normal; font-size: 16px;'>"
        "Le graphique des barres SHAP (SHAP bar plot) montre les features les plus importantes dans la prédiction du modèle. Chaque barre représente la valeur SHAP moyenne pour une feature donnée."
        "</p>",
        unsafe_allow_html=True
        )
        st_shap(shap.plots.bar(shap_values), height=300, width = 1100)
    with col2:
        st.markdown(
            "<h4 style='font-weight: normal; font-size: 24px;'>Graphique en essaim SHAP :</h4>", 
            unsafe_allow_html=True
        )
        st.markdown(
        "<p style='font-weight: normal; font-size: 16px;'>"
        "Ce graphique (SHAP beeswarm plot) permet de visualiser l'effet de chaque feature sur la prédiction du modèle pour chaque instance de données, ainsi que la distribution des valeurs SHAP pour chaque feature."
        "</p>",
        unsafe_allow_html=True
        )
        st_shap(shap.plots.beeswarm(shap_values), height=300, width = 900)

    col1, col2 = st.columns(2)    
    with col1:
        st.markdown(
            "<h4 style='font-weight: normal; font-size: 24px;'>Boites à moustaches des top 5 features :</h4>", 
            unsafe_allow_html=True
        )
        st.markdown(
        "<p style='font-weight: normal; font-size: 16px;'>"
        "Le box plot des top 5 features représente la distribution des valeurs normalisées de ces variables. Chaque ligne représente une variable, et la boîte indique le quartile de la distribution. Les points rouges représentent les valeurs du candidat sélectionné."
        "</p>",
        unsafe_allow_html=True
        )
        box_plot(top_features, selected_index)
    
    col1, col2 = st.columns((3,2))    
    with col1:
        st.markdown(
            "<h4 style='font-weight: normal; font-size: 24px;'>Glossaire des features les plus importantes :</h4>", 
            unsafe_allow_html=True
        )
        st.dataframe(glossaire, hide_index=True, use_container_width=True)









     
