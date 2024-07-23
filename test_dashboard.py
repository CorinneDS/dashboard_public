import pytest
import pandas as pd
import json
import requests
from unittest.mock import patch
from dashboard_prod import load_data, select_candidat, prediction 

# Définir la fonction random_test_data dans le script de test
def random_test_data():
    # Lire le fichier CSV
    df = pd.read_csv('X_test50_woIndex.csv')
    # Sélectionner une ligne au hasard
    random_row = df.sample(n=1).to_dict(orient='records')[0]
    return random_row

@pytest.fixture
def sample_data():
    # Charger les données d'échantillon
    return pd.read_csv('X_test50_withIndex.csv')

def test_load_data(sample_data):
    base_dir = '.'  # Assure-toi que ce chemin est correct pour ton environnement de test
    data = load_data(base_dir)
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert 'SK_ID_CURR' in data.columns


def test_select_candidat(sample_data):
    # Utiliser le patch pour simuler la sélection dans Streamlit
    with patch('streamlit.sidebar.selectbox', return_value=sample_data['SK_ID_CURR'].iloc[0]):
        with patch.dict('dashboard_prod.__dict__', {'data': sample_data}):
            candidat = select_candidat()

            # Vérifier le format de la réponse
            assert isinstance(candidat, dict)
            assert 'amt_annuity' in candidat  # Assure-toi que ces colonnes existent dans tes données
            assert 'amt_credit' in candidat

def test_prediction():
    row = random_test_data()
    prediction_values = prediction(row)

    # Vérifier que les valeurs de prédiction sont correctes
    assert len(prediction_values) == 2
    assert 0 <= prediction_values[0][0] <= 1
    assert 0 <= prediction_values[0][1] <= 1
    assert 0 <= prediction_values[1] <= 1
