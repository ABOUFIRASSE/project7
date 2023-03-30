
"""Projet 7 OCR: Implémentez un modèle de scoring
َAuteur: Badreddine ABOUFARASSE 
links: 
    https://www.linkedin.com/in/aboufarasse-badr-ab711018/
    https://www.bse.u-bordeaux.fr/en/membres/badr-aboufarasse/
    
Source: 
    https://github.com/ABOUFIRASSE/OCR_P7_Dashboard
    
Local URL: http://localhost:8501
Network URL: http://192.168.1.20:8501

"""

# ====================================================================
# Version : 0.0.2  B.ABOUFARASSE   03/2023
# ====================================================================

__version__ = '0.0.2'

# ====================================================================
# Chargement des librairies
# ====================================================================

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import shap
from pathlib import Path

# ====================================================================
# VARIABLES
# ====================================================================
# Best Model
BEST_MODELE_DATA = 'resources/modele/best_model.pickle'
# Application set
APPLICATION_TEST_DATA = 'resources/data/application_test.pickle'
# Test set
TEST_SET_DATA = 'resources/data/test_set.pickle'
# Dashboard data
DASHBOARD_DATA = 'resources/data/df_dashboard.pickle'
# Info clients et prêts 
CLIENT_INFO_DATA = 'resources/data/df_info_client.pickle'
CLIENT_PRET_DATA = 'resources/data/df_pret_client.pickle'
# 10 nearest neighbors
VOISINS_INFO_DATA = 'resources/data/df_info_voisins.pickle'
VOISIN_PRET_DATA = 'resources/data/df_pret_voisins.pickle'
VOISIN_AGG_DATA = 'resources/data/df_voisin_train_agg.pickle'
ALL_TRAIN_AGG_DATA = 'resources/data/df_all_train_agg.pickle'
# ====================================================================
# GLOBAL FEATURES
# ====================================================================
group_val1 = ['AMT_ANNUITY',
              'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN',
              'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN',
              'BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN',
              'INST_PAY_AMT_INSTALMENT_SUM']
group_val2 = ['CAR_EMPLOYED_RATIO', 'CODE_GENDER',
              'CREDIT_ANNUITY_RATIO', 'CREDIT_GOODS_RATIO',
              'YEAR_BIRTH', 'YEAR_ID_PUBLISH',
              'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
              'EXT_SOURCE_MAX', 'EXT_SOURCE_SUM',
              'FLAG_OWN_CAR',
              'INST_PAY_DAYS_PAYMENT_RATIO_MAX',
              'POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM',
              'PREV_APP_INTEREST_SHARE_MAX']
group_val3 = ['AMT_ANNUITY_MEAN',
              'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN_MEAN',
              'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN_MEAN',
              'BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN_MEAN',
              'INST_PAY_AMT_INSTALMENT_SUM_MEAN']


group_val4 = ['CAR_EMPLOYED_RATIO_MEAN', 'CODE_GENDER_MEAN',
              'CREDIT_ANNUITY_RATIO_MEAN', 'CREDIT_GOODS_RATIO_MEAN',
              'YEAR_BIRTH_MEAN', 'YEAR_ID_PUBLISH_MEAN',
              'EXT_SOURCE_1_MEAN', 'EXT_SOURCE_2_MEAN', 'EXT_SOURCE_3_MEAN',
              'EXT_SOURCE_MAX_MEAN', 'EXT_SOURCE_SUM_MEAN',
              'FLAG_OWN_CAR_MEAN',
              'INST_PAY_DAYS_PAYMENT_RATIO_MAX_MEAN',
              'POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM_MEAN',
              'PREV_APP_INTEREST_SHARE_MAX_MEAN']
# ====================================================================
# IMAGES
# ====================================================================
# Logo de l'entreprise
logo =  Image.open("resources/images/logoo.png") 
# Légende des courbes
lineplot_legende =  Image.open("resources/images/lineplot_legende.png") 

# ====================================================================
# HTML MARKDOWN
# ====================================================================
html_AMT_ANNUITY = "<h4 style='text-align: center'>AMT_ANNUITY</h4> <br/> <h5 style='text-align: center'>Annuité du prêt</h5> <hr/>"
html_BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN = "<h4 style='text-align: center'>BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN</h4> <br/> <h5 style='text-align: center'>Valeur minimale de la différence entre la limite de crédit actuelle de la carte de crédit et la dette actuelle sur le crédit</h5> <hr/>" 
html_BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN = "<h4 style='text-align: center'>BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN</h4> <br/> <h5 style='text-align: center'>Valeur moyenne de la différence entre la limite de crédit actuelle de la carte de crédit et la dette actuelle sur le crédit</h5> <hr/>" 
html_INST_PAY_AMT_INSTALMENT_SUM = "<h4 style='text-align: center'>INST_PAY_AMT_INSTALMENT_SUM</h4> <br/> <h5 style='text-align: center'>Somme du montant de l'acompte prescrit des crédits précédents sur cet acompte</h5> <hr/>" 
html_BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN = "<h4 style='text-align: center'>BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN</h4> <br/> <h5 style='text-align: center'>Moyenne du ratio des prêts précédents sur d'autres institution de : la dette actuelle sur le crédit et la limite de crédit actuelle de la carte de crédit (valeur * 100)</h5> <hr/>" 
html_CAR_EMPLOYED_RATIO = "<h4 style='text-align: center'>CAR_EMPLOYED_RATIO</h4> <br/> <h5 style='text-align: center'>Ratio : Âge de la voiture du demandeur / Ancienneté dans l'emploi à la date de la demande (valeur * 1000)</h5> <hr/>" 
html_CODE_GENDER = "<h4 style='text-align: center'>CODE_GENDER</h4> <br/> <h5 style='text-align: center'>Sexe</h5> <hr/>" 
html_CREDIT_ANNUITY_RATIO = "<h4 style='text-align: center'>CREDIT_ANNUITY_RATIO</h4> <br/> <h5 style='text-align: center'>Ratio : montant du crédit du prêt / Annuité de prêt</h5> <hr/>" 
html_CREDIT_GOODS_RATIO = "<h4 style='text-align: center'>CREDIT_GOODS_RATIO</h4> <br/> <h5 style='text-align: center'>Ratio : Montant du crédit du prêt / prix des biens pour lesquels le prêt est accordé / Crédit est supérieur au prix des biens ? (valeur * 100)</h5> <hr/>" 
html_YEAR_BIRTH = "<h4 style='text-align: center'>YEAR_BIRTH</h4> <br/> <h5 style='text-align: center'>Âge (ans)</h5> <hr/>" 
html_YEAR_ID_PUBLISH = "<h4 style='text-align: center'>YEAR_ID_PUBLISH</h4> <br/> <h5 style='text-align: center'>Combien de jours avant la demande le client a-t-il changé la pièce d'identité avec laquelle il a demandé le prêt ? (ans)</h5> <hr/>" 
html_EXT_SOURCE_1 = "<h4 style='text-align: center'>EXT_SOURCE_1</h4> <br/> <h5 style='text-align: center'>Source externe normalisée (valeur * 100)</h5> <hr/>" 
html_EXT_SOURCE_2 = "<h4 style='text-align: center'>EXT_SOURCE_2</h4> <br/> <h5 style='text-align: center'>Source externe normalisée (valeur * 100)</h5> <hr/>" 
html_EXT_SOURCE_3 = "<h4 style='text-align: center'>EXT_SOURCE_3</h4> <br/> <h5 style='text-align: center'>Source externe normalisée (valeur * 100)</h5> <hr/>" 
html_EXT_SOURCE_MAX = "<h4 style='text-align: center'>EXT_SOURCE_MAX</h4> <br/> <h5 style='text-align: center'>Valeur maximale des 3 sources externes normalisées (EXT_SOURCE_1, EXT_SOURCE_2 et EXT_SOURCE_3) (valeur * 100)</h5> <hr/>" 
html_EXT_SOURCE_SUM = "<h4 style='text-align: center'>EXT_SOURCE_SUM</h4> <br/> <h5 style='text-align: center'>Somme des 3 sources externes normalisées (EXT_SOURCE_1, EXT_SOURCE_2 et EXT_SOURCE_3, valeur * 100)</h5> <hr/>" 
html_FLAG_OWN_CAR = "<h4 style='text-align: center'>FLAG_OWN_CAR</h4> <br/> <h5 style='text-align: center'>Indicateur si le client possède une voiture</h5> <hr/>" 
html_INST_PAY_DAYS_PAYMENT_RATIO_MAX = "<h4 style='text-align: center'>INST_PAY_DAYS_PAYMENT_RATIO_MAX</h4> <br/> <h5 style='text-align: center'>Valeur maximal dans l'historique des précédents crédits remboursés dans Home Crédit du ratio : La date à laquelle le versement du crédit précédent était censé être payé (par rapport à la date de demande du prêt actuel) \ Quand les échéances du crédit précédent ont-elles été effectivement payées (par rapport à la date de demande du prêt</h5> <hr/>" 
html_NAME_EDUCATION_TYPE_HIGHER_EDUCATION = "<h4 style='text-align: center'>NAME_EDUCATION_TYPE_HIGHER_EDUCATION</h4> <br/> <h5 style='text-align: center'>Niveau d'éducation le plus élévé</h5> <hr/>" 
html_POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM = "<h4 style='text-align: center'>POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM</h4> <br/> <h5 style='text-align: center'>Somme des contrats actifs au cours du mois</h5> <hr/>" 
html_PREV_APP_INTEREST_SHARE_MAX = "<h4 style='text-align: center'>PREV_APP_INTEREST_SHARE_MAX</h4> <br/> <h5 style='text-align: center'>La valeur maximale de tous les précédents crédit dans d'autres institution : de la durée du crédit multiplié par l'annuité du prêt moins le montant final du crédit</h5> <hr/>" 

# ====================================================================
# TITRE
# ====================================================================
html_header="""
    <head>
        <title>Application Dashboard Crédit Score</title>
        <meta charset="utf-8">
        <meta name="keywords" content="Home Crédit Group, Dashboard, prêt, crédit score">
        <meta name="description" content="Application de Crédit Score - dashboard">
        <meta name="author" content="Badr ABOUFARASSE">
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>             
    <h1 style="font-size:350%; color:White; font-family:Times new roman"> CREDIT SCORING <br>
        <h2 style="color:Gray; font-family:GREY"> DASHBOARD CLIENT NOTATION</h2>
        <hr style= "  display: block;
          margin-top: 0;
          margin-bottom: 0;
          margin-left: center;
          margin-right: center;
          border-style: inset;
          border-width: 2.5px;"/>
     </h1>   
"""
st.set_page_config(page_title="CREDIT SCORING - DACHBOARD CLIENT SCORING", page_icon="", layout="wide")
st.markdown('<style>body{background-color: #fbfff0}</style>',unsafe_allow_html=True)
st.markdown(html_header, unsafe_allow_html=True)

# Cacher le bouton en haut à droite
st.markdown(""" <style>
MainMenu {visibility: visible;}
footer {visibility: visible;}
</style> """, unsafe_allow_html=True)

# Suppression des marges par défaut

padding = 1
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

# ====================================================================
# DONNEES
# ====================================================================
# Chargement du modèle et des différents dataframes
@st.cache (persist = False)
def load():
    with st.spinner('Import des données'):        
        # Import du dataframe des informations des traits stricts du client
        fic_client_info = CLIENT_INFO_DATA
        with open(fic_client_info, 'rb') as df_info_client:
            df_info_client = pickle.load(df_info_client)                
        # Import du dataframe des informations sur le prêt du client
        fic_client_pret = CLIENT_PRET_DATA
        with open(fic_client_pret, 'rb') as df_pret_client:
            df_pret_client = pickle.load(df_pret_client)            
        # Import du dataframe des informations des traits stricts des voisins
        fic_voisin_info = VOISINS_INFO_DATA
        with open(fic_voisin_info, 'rb') as df_info_voisins:
            df_info_voisins = pickle.load(df_info_voisins)          
        # Import du dataframe des informations sur le prêt des voisins
        fic_voisin_pret = VOISIN_PRET_DATA
        with open(fic_voisin_pret, 'rb') as df_pret_voisins:
            df_pret_voisins = pickle.load(df_pret_voisins)
        # Import du dataframe des informations sur le dashboard
        fic_dashboard = DASHBOARD_DATA
        with open(fic_dashboard, 'rb') as df_dashboard:
            df_dashboard = pickle.load(df_dashboard)
        # Import du dataframe des informations sur les voisins aggrégés
        fic_voisin_train_agg = VOISIN_AGG_DATA
        with open(fic_voisin_train_agg, 'rb') as df_voisin_train_agg:
            df_voisin_train_agg = pickle.load(df_voisin_train_agg)
        # Import du dataframe des informations sur les voisins aggrégés
        fic_all_train_agg = ALL_TRAIN_AGG_DATA
        with open(fic_all_train_agg, 'rb') as df_all_train_agg:
            df_all_train_agg = pickle.load(df_all_train_agg)
        # Import du dataframe des informations sur les voisins aggrégés
        with open(TEST_SET_DATA, 'rb') as df_test_set:
            test_set = pickle.load(df_test_set)
        # Import du dataframe du test set brut original
        with open(APPLICATION_TEST_DATA, 'rb') as df_application_test:
            application_test = pickle.load(df_application_test)                    
    # Import du meilleur modèle lgbm entrainé
    with st.spinner('Import du modèle'):        
        # Import du meilleur modèle lgbm entrainé
        fic_best_model = BEST_MODELE_DATA
        with open(fic_best_model, 'rb') as model_lgbm:
            best_model = pickle.load(model_lgbm)
    # SHAP values
    with st.spinner('Lancement SHAP values'):
        # Test set sans l'identifiant
        X_bar = test_set.set_index('SK_ID_CURR')
        # Entraînement de shap sur le train set
        bar_explainer = shap.Explainer(best_model, X_bar)
        bar_values = bar_explainer(X_bar, check_additivity=False)              
    return df_info_client, df_pret_client, df_info_voisins, df_pret_voisins, \
        df_dashboard, df_voisin_train_agg, df_all_train_agg, test_set, \
            application_test, best_model, bar_values
# Chargement des dataframes et du modèle
df_info_client, df_pret_client, df_info_voisins, df_pret_voisins, \
    df_dashboard, df_voisin_train_agg, df_all_train_agg, test_set, \
            application_test, best_model, bar_values = load()
# ====================================================================
# How to use
# ====================================================================
with st.expander("How to Use this Platform"):
    st.write(Path("Readme.md").read_text())
# ====================================================================
# CHOIX DU CLIENT
# ====================================================================
html_select_client="""
    <div class="card">
      <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #DEC7CB; padding-top: 5px; width: auto;
                  height: 40px;">
        <h3 class="card-title" style="background-color:#DEC7CB; color:Crimson;
                   font-family:Georgia; text-align: center; padding: 0px 0;">
          IDENTIFICATION CLIENT DEMANDEUR
        </h3>
      </div>
    </div>
    """
st.markdown(html_select_client, unsafe_allow_html=True)
with st.expander("Détail Info clients"):
    st.write(df_info_client)
with st.container():
    col1, col2 = st.columns([1,4])
    with col1:
        st.write("")
        col1.header("**ID Client**")
        client_id = col1.selectbox('Client :',
                                   df_info_voisins['ID_CLIENT'].unique())
    
with st.container():
    col1, col2 = st.columns([0.1,6])
    with col2:
        # Infos principales client
        st.write("*Caractéristiques clients*")
        client_info = df_info_client[df_info_client['SK_ID_CURR'] == client_id].iloc[:, :]
        client_info.set_index('SK_ID_CURR', inplace=True)
        st.table(client_info)
        # Infos principales sur la demande de prêt
        st.write("*Historique des Demandes de prêt*")
        client_pret = df_pret_client[df_pret_client['SK_ID_CURR'] == client_id].iloc[:, :]
        client_pret.set_index('SK_ID_CURR', inplace=True)
        st.table(client_pret)

st.subheader("Upload your file")
uploaded_data = st.file_uploader(
    "Drag and Drop or Click to Upload", type=".csv", accept_multiple_files=True
)
    
# ====================================================================
# SCORE - PREDICTIONS
# ====================================================================

html_score="""
    <div class="card">
      <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #DEC7CB; padding-top: 5px; width: auto;
                  height: 40px;">
        <h3 class="card-title" style="background-color:#DEC7CB; color:Crimson;
                   font-family:Georgia; text-align: center; padding: 0px 0;">
          SCORE CLIENT
        </h3>
      </div>
    </div>
    """

st.markdown(html_score, unsafe_allow_html=True)

# Préparation des données à afficher dans la jauge ==============================================

# ============== Score du client en pourcentage ==> en utilisant le modèle ======================
# Sélection des variables du clients
X_test = test_set[test_set['SK_ID_CURR'] == client_id]
# Score des prédictions de probabiltés
y_proba = best_model.predict_proba(X_test.drop('SK_ID_CURR', axis=1))[:, 1]
# Score du client en pourcentage arrondi et nombre entier
score_client = int(np.rint(y_proba * 100))

# ============== Score moyen des 10 plus proches voisins du test set en pourcentage =============

# Score moyen des 10 plus proches voisins du test set en pourcentage
score_moy_voisins_test = int(np.rint(df_dashboard[
    df_dashboard['SK_ID_CURR'] == client_id]['SCORE_10_VOISINS_MEAN_TEST'] * 100))

# ============== Pourcentage de clients voisins défaillants dans l'historique des clients =======
pourc_def_voisins_train = int(np.rint(df_dashboard[
    df_dashboard['SK_ID_CURR'] == client_id]['%_NB_10_VOISINS_DEFAILLANT_TRAIN']))

# ============== Pourcentage de clients voisins défaillants prédits parmi les nouveaux clients ==
pourc_def_voisins_test = int(np.rint(df_dashboard[
    df_dashboard['SK_ID_CURR'] == client_id]['%_NB_10_VOISINS_DEFAILLANT_TEST']))


# Graphique de jauge du cédit score ========================================== #https://dash.plotly.com/dash-daq/darkthemeprovider
fig_jauge = go.Figure(go.Indicator(
    mode = 'gauge+number+delta',
    # Score du client en % df_dashboard['SCORE_CLIENT_%']
    value = score_client,  
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': 'SCORE', 'font': {'size': 45}},
    # Score des 10 voisins test set
    # df_dashboard['SCORE_10_VOISINS_MEAN_TEST']
    delta = {'reference': score_moy_voisins_test,
             'increasing': {'color': 'Red'},
             'decreasing': {'color': 'Green'}},
    
    gauge = {'axis': {'range': [None, 100],
                      'tickwidth': 10,
                      'tickcolor': 'darkblue'},
             'bar': {'color': 'black', 'thickness' : 0.35},
             'bgcolor': 'white',
             'borderwidth': 3.5,
             'bordercolor': 'blue',
             
             'steps': [{'range': [0, 20], 'color': 'lime'},
                       {'range': [20, 40], 'color': 'turquoise'},
                       {'range': [40, 60], 'color': 'lightseagreen'},
                       {'range': [60, 80], 'color': 'aqua'},
                       {'range': [80, 100], 'color': 'deepskyblue'}],
             
             
             'threshold': {'line': {'color': 'yellow', 'width': 7},
                           'thickness': 0.8,
                           # Score du client en %
                           # df_dashboard['SCORE_CLIENT_%']
                           'value': score_client}}))

fig_jauge.update_layout(paper_bgcolor='snow',
                        height=400, width=1000,
                        font={'color': 'black', 'family': 'Comic'},
                        margin=dict(l=6, r=6, b=4, t=4, pad=4))

# =============================================================================
st.success('This is a success message!', icon="✅")


with st.container():
    # JAUGE + récapitulatif du score moyen des voisins
    col1, col2 = st.columns([5, 0.00001])
    with col1:
        st.plotly_chart(fig_jauge)
    # with col2:
    #     st.write("")
    #     st.write("")
    #     st.write("")
    #     st.write("")
    #     st.write("")
    #     # Texte d'accompagnement de la jauge
    #     if 0 <= score_client < 25:
    #         score_text = 'Crédit score : EXCELLENT - Crédit accordé'
    #         st.success(score_text)
    #     elif 25 <= score_client < 50:
    #         score_text = 'Crédit score : BON - Crédit accordé'
    #         st.success(score_text)
    #     elif 50 <= score_client < 75:
    #         score_text = 'Crédit score : MOYEN - Crédit accordé sous conditions'
    #         st.warning(score_text)
    #     else :
    #         score_text = 'Crédit score : BAS - Crédit non accordé'
    #         st.error(score_text)
    #     st.write("")    
    #     # st.markdown(f'Crédit score moyen des 10 clients similaires : **{score_moy_voisins_test}**')
    #     # st.markdown(f'**{pourc_def_voisins_train}**% de clients voisins réellement défaillants dans l\'historique')
    #     # st.markdown(f'**{pourc_def_voisins_test}**% de clients voisins défaillants prédits pour les nouveaux clients')
   

with st.container():
    # JAUGE + récapitulatif du score moyen des voisins
    col1, col2 = st.columns([0.1,8])
    # with col1:
    #     st.plotly_chart(fig_jauge)
    with col2:
        st.write("")
        # st.write("")
        # st.write("")
        # st.write("")
        # st.write("")
        # Texte d'accompagnement de la jauge
        if 0 <= score_client < 25:
            score_text = 'Crédit score : EXCELLENT - Crédit accordé'
            st.success(score_text)
        elif 25 <= score_client < 50:
            score_text = 'Crédit score : BON - Crédit accordé'
            st.success(score_text)
        elif 50 <= score_client < 75:
            score_text = 'Crédit score : MOYEN - Crédit accordé sous conditions'
            st.warning(score_text)
        else :
            score_text = 'Crédit score : BAS - Crédit non accordé'
            st.error(score_text)
        st.write("")    
        # st.markdown(f'Crédit score moyen des 10 clients similaires : **{score_moy_voisins_test}**')
        # st.markdown(f'**{pourc_def_voisins_train}**% de clients voisins réellement défaillants dans l\'historique')
        # st.markdown(f'**{pourc_def_voisins_test}**% de clients voisins défaillants prédits pour les nouveaux clients')
   

# ====================================================================
# SIDEBAR
# ====================================================================

# Toutes Les informations non modifiées du client courant
df_client_origin = application_test[application_test['SK_ID_CURR'] == client_id]

# Toutes Les informations non modifiées du client courant
df_client_test = test_set[test_set['SK_ID_CURR'] == client_id]

# Toutes les informations du client courant
df_client_courant = df_dashboard[df_dashboard['SK_ID_CURR'] == client_id]


# --------------------------------------------------------------------
# LOGO
# --------------------------------------------------------------------
# Chargement du logo de l'entreprise
st.sidebar.image(logo, width=200, caption="",
                use_column_width='always')


# --------------------------------------------------------------------
# PLUS INFORMATIONS
# --------------------------------------------------------------------
def all_infos_clients():
    ''' Affiche toutes les informations sur le client courant
    '''
    html_all_infos_clients="""
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #DEC7CB; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:#DEC7CB; color:Crimson;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                      INFORMATIONS CLIENTS
                  </h3>
            </div>
        </div>
        """
    
    # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES =========================== 
    if st.sidebar.checkbox("Fiche client"):     
        
        st.markdown(html_all_infos_clients, unsafe_allow_html=True)

        with st.spinner('**Affiche toutes les informations sur le client courant...**'):                 
                       
            with st.expander('Amples informations sur le client identifié',
                             expanded=False):

                st.dataframe(df_client_origin)

                st.dataframe(df_client_test)

st.sidebar.subheader('Détails client')

all_infos_clients()

# --------------------------------------------------------------------
# CLIENTS SIMILAIRES 
# --------------------------------------------------------------------
def infos_clients_similaires():
    ''' Affiche les informations sur les clients similaires :
            - traits stricts.
            - demande de prêt
    '''
    html_clients_similaires="""
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #DEC7CB; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:#DEC7CB; color:Crimson;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                      Clients similaires
                  </h3>
            </div>
        </div>
        """
    
    titre = True

    # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES =========================== 
    if st.sidebar.checkbox("Voir graphiques comparatifs ?"):     
        
        if titre:
            st.markdown(html_clients_similaires, unsafe_allow_html=True)
            titre = False

        with st.spinner('**Affiche les graphiques comparant le client courant et les clients similaires...**'):                 
                       
            with st.expander('Comparaison variables impactantes client courant/moyennes des clients similaires',
                             expanded=True):
                with st.container():
                    # Préparatifs dataframe
                    df_client = df_voisin_train_agg[df_voisin_train_agg['ID_CLIENT'] == client_id].astype(int)
                    # ====================================================================
                    # Lineplot comparatif features importances client courant/voisins
                    # ====================================================================
                    
                    # ===================== Valeurs moyennes des features importances pour le client courant =====================
                    df_feat_client  = df_client_courant[['SK_ID_CURR', 'AMT_ANNUITY',
                               'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN',
                               'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN',
                               'BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN',
                               'CAR_EMPLOYED_RATIO', 'CODE_GENDER',
                               'CREDIT_ANNUITY_RATIO', 'CREDIT_GOODS_RATIO',
                               'DAYS_BIRTH', 'DAYS_ID_PUBLISH',
                               'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                               'EXT_SOURCE_MAX', 'EXT_SOURCE_SUM',
                               'FLAG_OWN_CAR', 'INST_PAY_AMT_INSTALMENT_SUM',
                               'INST_PAY_DAYS_PAYMENT_RATIO_MAX',
                               'POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM',
                               'PREV_APP_INTEREST_SHARE_MAX']]
                    df_feat_client['YEAR_BIRTH'] = \
                        np.trunc(np.abs(df_feat_client['DAYS_BIRTH'] / 365)).astype('int8')
                    df_feat_client['YEAR_ID_PUBLISH'] = \
                        np.trunc(np.abs(df_feat_client['DAYS_ID_PUBLISH'] / 365)).astype('int8')
                    df_feat_client.drop(columns=['DAYS_BIRTH', 'DAYS_ID_PUBLISH'],
                                        inplace=True)
                    df_feat_client_gp1 = df_feat_client[group_val1]
                    df_feat_client_gp2 = df_feat_client[group_val2]
                    # X
                    x_gp1 = df_feat_client_gp1.columns.to_list()
                    x_gp2 = df_feat_client_gp2.columns.to_list()
                    # y
                    y_feat_client_gp1 = df_feat_client_gp1.values[0].tolist()
                    y_feat_client_gp2 = df_feat_client_gp2.values[0].tolist()
                    
                    # ===================== Valeurs moyennes des features importances pour les 10 voisins =======================
                    df_moy_feat_voisins = df_client[['ID_CLIENT', 'AMT_ANNUITY_MEAN',
                               'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN_MEAN',
                               'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN_MEAN',
                               'BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN_MEAN',
                               'CAR_EMPLOYED_RATIO_MEAN', 'CODE_GENDER_MEAN',
                               'CREDIT_ANNUITY_RATIO_MEAN', 'CREDIT_GOODS_RATIO_MEAN',
                               'DAYS_BIRTH_MEAN', 'DAYS_ID_PUBLISH_MEAN',
                               'EXT_SOURCE_1_MEAN', 'EXT_SOURCE_2_MEAN', 'EXT_SOURCE_3_MEAN',
                               'EXT_SOURCE_MAX_MEAN', 'EXT_SOURCE_SUM_MEAN',
                               'FLAG_OWN_CAR_MEAN', 'INST_PAY_AMT_INSTALMENT_SUM_MEAN',
                               'INST_PAY_DAYS_PAYMENT_RATIO_MAX_MEAN',
                               'POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM_MEAN',
                               'PREV_APP_INTEREST_SHARE_MAX_MEAN']]
                    df_moy_feat_voisins['YEAR_BIRTH_MEAN'] = \
                        np.trunc(np.abs(df_moy_feat_voisins['DAYS_BIRTH_MEAN'] / 365)).astype('int8')
                    df_moy_feat_voisins['YEAR_ID_PUBLISH_MEAN'] = \
                        np.trunc(np.abs(df_moy_feat_voisins['DAYS_ID_PUBLISH_MEAN'] / 365)).astype('int8')
                    df_moy_feat_voisins.drop(columns=['DAYS_BIRTH_MEAN', 'DAYS_ID_PUBLISH_MEAN'],
                                        inplace=True)
                    df_moy_feat_voisins_gp3 = df_moy_feat_voisins[group_val3]
                    df_moy_feat_voisins_gp4 = df_moy_feat_voisins[group_val4]
                    # y
                    y_moy_feat_voisins_gp3 = df_moy_feat_voisins_gp3.values[0].tolist()
                    y_moy_feat_voisins_gp4 = df_moy_feat_voisins_gp4.values[0].tolist()
                    
                    # ===================== Valeurs moyennes de tous les clients non-défaillants/défaillants du train sets =======================
                    df_all_train = df_all_train_agg[['TARGET', 'AMT_ANNUITY_MEAN',
                               'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN_MEAN',
                               'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN_MEAN',
                               'BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN_MEAN',
                               'CAR_EMPLOYED_RATIO_MEAN', 'CODE_GENDER_MEAN',
                               'CREDIT_ANNUITY_RATIO_MEAN', 'CREDIT_GOODS_RATIO_MEAN',
                               'YEAR_BIRTH_MEAN', 'DAYS_ID_PUBLISH_MEAN',
                               'EXT_SOURCE_1_MEAN', 'EXT_SOURCE_2_MEAN', 'EXT_SOURCE_3_MEAN',
                               'EXT_SOURCE_MAX_MEAN', 'EXT_SOURCE_SUM_MEAN',
                               'FLAG_OWN_CAR_MEAN', 'INST_PAY_AMT_INSTALMENT_SUM_MEAN',
                               'INST_PAY_DAYS_PAYMENT_RATIO_MAX_MEAN',
                               'POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM_MEAN',
                               'PREV_APP_INTEREST_SHARE_MAX_MEAN']]
                    df_all_train['YEAR_ID_PUBLISH_MEAN'] = \
                        np.trunc(np.abs(df_all_train['DAYS_ID_PUBLISH_MEAN'] / 365)).astype('int8')
                    df_all_train.drop(columns=['DAYS_ID_PUBLISH_MEAN'],
                                        inplace=True)
                    # Non-défaillants
                    df_all_train_nondef_gp3 = df_all_train[df_all_train['TARGET'] == 0][group_val3]
                    df_all_train_nondef_gp4 = df_all_train[df_all_train['TARGET'] == 0][group_val4]
                    # Défaillants
                    df_all_train_def_gp3 = df_all_train[df_all_train['TARGET'] == 1][group_val3]
                    df_all_train_def_gp4 = df_all_train[df_all_train['TARGET'] == 1][group_val4]
                    # y
                    # Non-défaillants
                    y_all_train_nondef_gp3 = df_all_train_nondef_gp3.values[0].tolist()
                    y_all_train_nondef_gp4 = df_all_train_nondef_gp4.values[0].tolist()
                    # Défaillants
                    y_all_train_def_gp3 = df_all_train_def_gp3.values[0].tolist()
                    y_all_train_def_gp4 = df_all_train_def_gp4.values[0].tolist()

                    # Légende des courbes
                    st.image(lineplot_legende)
                                                  
                    col1, col2 = st.columns([1, 1.5])
                    with col1:
                        # Lineplot de comparaison des features importances client courant/voisins/all ================
                        plt.figure(figsize=(6, 6))
                        plt.plot(x_gp1, y_feat_client_gp1, color='Orange')
                        plt.plot(x_gp1, y_moy_feat_voisins_gp3, color='SteelBlue')
                        plt.plot(x_gp1, y_all_train_nondef_gp3, color='Green')
                        plt.plot(x_gp1, y_all_train_def_gp3, color='Crimson')
                        plt.xticks(rotation=90)
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot()
                    with col2: 
                        # Lineplot de comparaison des features importances client courant/voisins/all ================
                        plt.figure(figsize=(8, 5))
                        plt.plot(x_gp2, y_feat_client_gp2, color='Orange')
                        plt.plot(x_gp2, y_moy_feat_voisins_gp4, color='SteelBlue')
                        plt.plot(x_gp2, y_all_train_nondef_gp4, color='Green')
                        plt.plot(x_gp2, y_all_train_def_gp4, color='Crimson')
                        plt.xticks(rotation=90)
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot()
                        
                    with st.container(): 
                        
                        vars_select = ['AMT_ANNUITY', 
                                       'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN',
                                       'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN',
                                       'BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN',
                                       'CAR_EMPLOYED_RATIO', 
                                       'CODE_GENDER',
                                       'CREDIT_ANNUITY_RATIO',
                                       'CREDIT_GOODS_RATIO',
                                       'EXT_SOURCE_1', 
                                       'EXT_SOURCE_2', 
                                       'EXT_SOURCE_3',
                                       'EXT_SOURCE_MAX', 
                                       'EXT_SOURCE_SUM',
                                       'FLAG_OWN_CAR',
                                       'INST_PAY_AMT_INSTALMENT_SUM',
                                       'INST_PAY_DAYS_PAYMENT_RATIO_MAX',
                                       'NAME_EDUCATION_TYPE_HIGHER_EDUCATION',
                                       'POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM',
                                       'PREV_APP_INTEREST_SHARE_MAX',
                                       'YEAR_BIRTH', 
                                       'YEAR_ID_PUBLISH']

                        feat_imp_to_show = st.multiselect("Feature(s) importance(s) à visualiser : ",
                                                          vars_select)

                        # ==============================================================
                        # Variable AMT_ANNUITY
                        # Annuité du prêt
                        # ==============================================================

                        if 'AMT_ANNUITY' in feat_imp_to_show:
                            
                            with st.spinner('**Chargement du graphique comparatif AMT_ANNUITY...**'):

                                amt_min = int(df_client.AMT_ANNUITY_MIN.values)
                                amt_q25 = int(df_client.AMT_ANNUITY_Q25.values)
                                amt_mean = int(df_client.AMT_ANNUITY_MEAN.values)
                                amt_q75 = int(df_client.AMT_ANNUITY_Q75.values)
                                amt_max = int(df_client.AMT_ANNUITY_MAX.values)
                                amt_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                    'AMT_ANNUITY'].values)
                                amt_axis_min = min(amt_min, amt_client)
                                amt_axis_max = max(amt_max, amt_client)
                                
                                fig_amt = go.Figure()
                                
                                fig_amt.add_trace(go.Indicator(
                                    mode = "number+gauge+delta",
                                    value = amt_client,
                                    delta = {'reference': amt_mean,
                                             'increasing': {'color': 'Crimson'},
                                             'decreasing': {'color': 'Green'}},
                                    domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                    title = {'text': 'Annuité', 'font': {'size': 12},
                                             'align' : 'left'},
                                    gauge = {
                                        'shape': 'bullet',
                                        'axis': {'range': [amt_axis_min, amt_axis_max]},
                                        'threshold': {
                                            'line': {'color': 'black', 'width': 3},
                                            'thickness': 0.75,
                                            'value': amt_client},
                                        'steps': [
                                            {'range': [0, amt_min], 'color': 'white'},
                                            {'range': [amt_min, amt_q25], 'color': '#de3a5b'},
                                            {'range': [amt_q25, amt_mean], 'color': '#dec7cb',
                                             'line': {'color': 'DarkSlateGray', 'width': 2}},
                                            {'range': [amt_mean, amt_q75],'color': '#dec7cb',
                                             'line': {'color': 'DarkSlateGray', 'width': 2}},
                                            {'range': [amt_q75, amt_max], 'color': '#de3a5b'}],
                                        'bar': {'color': 'black'}}))
                                
                                fig_amt.update_layout(height=200,
                                                      margin={'t':0, 'b':0, 'l':0})
                                
                                st.markdown(html_AMT_ANNUITY, unsafe_allow_html=True)

                                # ==================== Go Indicator bullets ==============================================
                                st.plotly_chart(fig_amt)

                                # ==================== ViolinPlot ========================================================
                                sns.violinplot(x='PRED_CLASSE_CLIENT', y='AMT_ANNUITY',
                                               data=df_dashboard,
                                               palette=['Green', 'Crimson'])

                                plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                         amt_client,
                                         color="orange",
                                         marker="$\\bigotimes$", markersize=28)
                                plt.xlabel('TARGET', fontsize=16)
                                client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                       linestyle='None',
                                                       markersize=16, label='Position du client')
                                plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                st.set_option('deprecation.showPyplotGlobalUse', False)
                                st.pyplot()
                                    
                                # ==================== DistPlot ==========================================================
                                # Non-défaillants
                                sns.distplot(df_dashboard['AMT_ANNUITY'][df_dashboard[
                                    'PRED_CLASSE_CLIENT'] == 0],
                                             label='Non-Défaillants', hist=False, color='Green')
                                # Défaillants
                                sns.distplot(df_dashboard['AMT_ANNUITY'][df_dashboard[
                                    'PRED_CLASSE_CLIENT'] == 1],
                                             label='Défaillants', hist=False, color='Crimson')
                                plt.xlabel('AMT_ANNUITY', fontsize=16)
                                plt.ylabel('Probability Density', fontsize=16)
                                plt.xticks(fontsize=16, rotation=90)
                                plt.yticks(fontsize=16)
                                # Position du client
                                plt.axvline(x=amt_client, color='orange', label='Position du client')
                                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                st.set_option('deprecation.showPyplotGlobalUse', False)
                                st.pyplot()                                  


                        # ==============================================================
                        # Variable BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN
                        # Valeur minimum de la différence entre la limite de crédit actuelle
                        # de la carte de crédit et la dette actuelle sur le crédit
                        # ==============================================================
                        if 'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN...**'):

                                bccddm_min = int(df_client.BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN_MIN.values)
                                bccddm_q25 = int(df_client.BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN_Q25.values)
                                bccddm_mean = int(df_client.BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN_MEAN.values)
                                bccddm_q75 = int(df_client.BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN_Q75.values)
                                bccddm_max = int(df_client.BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN_MAX.values)
                                bccddm_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                    'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN'].values)
                                bccddm_axis_min = min(bccddm_min, bccddm_client)
                                bccddm_axis_max = max(bccddm_max, bccddm_client)
                                
                                fig_bccdm = go.Figure()
                                
                                fig_bccdm.add_trace(go.Indicator(
                                    mode = "number+gauge+delta",
                                    value = bccddm_client,
                                    delta = {'reference': bccddm_mean,
                                             'increasing': {'color': 'Crimson'},
                                             'decreasing': {'color': 'Green'}},
                                    domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                    title = {'text': 'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN',
                                             'font': {'size': 9}, 'align' : 'left'},
                                    gauge = {
                                        'shape': 'bullet',
                                        'axis': {'range': [bccddm_axis_min, bccddm_axis_max]},
                                        'threshold': {
                                            'line': {'color': 'black', 'width': 3},
                                            'thickness': 0.75,
                                            'value': bccddm_client},
                                        'steps': [
                                            {'range': [0, bccddm_min], 'color': 'white'},
                                            {'range': [bccddm_min, bccddm_q25], 'color': '#de3a5b'},
                                            {'range': [bccddm_q25, bccddm_mean], 'color': '#dec7cb',
                                             'line': {'color': 'DarkSlateGray', 'width': 2}},
                                            {'range': [bccddm_mean, bccddm_q75],'color': '#dec7cb',
                                             'line': {'color': 'DarkSlateGray', 'width': 2}},
                                            {'range': [bccddm_q75, bccddm_max], 'color': '#de3a5b'}],
                                        'bar': {'color': 'black'}}))
                                
                                fig_bccdm.update_layout(height=200,
                                                        margin={'t':0, 'b':0, 'l':0})
                                
                                st.markdown(html_BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN, unsafe_allow_html=True)

                                # Go Indicator bullets
                                st.plotly_chart(fig_bccdm)           
                                
                                # ==================== ViolinPlot ========================================================
                                sns.violinplot(x='PRED_CLASSE_CLIENT', y='BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN',
                                               data=df_dashboard,
                                               palette=['Green', 'Crimson'])

                                plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                         bccddm_client,
                                         color="orange",
                                         marker="$\\bigotimes$", markersize=28)
                                plt.xlabel('TARGET', fontsize=16)
                                client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                       linestyle='None',
                                                       markersize=16, label='Position du client')
                                plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                st.set_option('deprecation.showPyplotGlobalUse', False)
                                st.pyplot()
                                    
                                # ==================== DistPlot ==========================================================
                                # Non-défaillants
                                sns.distplot(df_dashboard['BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN'][df_dashboard[
                                    'PRED_CLASSE_CLIENT'] == 0],
                                             label='Non-Défaillants', hist=False, color='Green')
                                # Défaillants
                                sns.distplot(df_dashboard['BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN'][df_dashboard[
                                    'PRED_CLASSE_CLIENT'] == 1],
                                             label='Défaillants', hist=False, color='Crimson')
                                plt.xlabel('BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN', fontsize=16)
                                plt.ylabel('Probability Density', fontsize=16)
                                plt.xticks(fontsize=16, rotation=90)
                                plt.yticks(fontsize=16)
                                # Position du client
                                plt.axvline(x=bccddm_client, color='orange', label='Position du client')
                                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                st.set_option('deprecation.showPyplotGlobalUse', False)
                                st.pyplot()                                  

                                
                        # ==============================================================
                        # Variable BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN
                        # Valeur moyenne de la différence entre la limite de crédit actuelle
                        # de la carte de crédit et la dette actuelle sur le crédit
                        # ==============================================================
                        if 'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN...**'):

                                bccddmean_min = int(df_client.BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN_MIN.values)
                                bccddmean_q25 = int(df_client.BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN_Q25.values)
                                bccddmean_mean = int(df_client.BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN_MEAN.values)
                                bccddmean_q75 = int(df_client.BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN_Q75.values)
                                bccddmean_max = int(df_client.BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN_MAX.values)
                                bccddmean_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                    'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN'].values)
                                bccddmean_axis_min = min(bccddmean_min, bccddmean_client)
                                bccddmean_axis_max = max(bccddmean_max, bccddmean_client)
                                
                                fig_bccddmean = go.Figure()
                                
                                fig_bccddmean.add_trace(go.Indicator(
                                    mode = "number+gauge+delta",
                                    value = bccddmean_client,
                                    delta = {'reference': bccddmean_mean,
                                             'increasing': {'color': 'Crimson'},
                                             'decreasing': {'color': 'Green'}},
                                    domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                    title = {'text': 'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN',
                                             'font': {'size': 9}, 'align' : 'left'},
                                    gauge = {
                                        'shape': 'bullet',
                                        'axis': {'range': [bccddmean_axis_min, bccddmean_axis_max]},
                                        'threshold': {
                                            'line': {'color': 'black', 'width': 3},
                                            'thickness': 0.75,
                                            'value': bccddmean_client},
                                        'steps': [
                                            {'range': [0, bccddmean_min], 'color': 'white'},
                                            {'range': [bccddmean_min, bccddmean_q25], 'color': '#de3a5b'},
                                            {'range': [bccddmean_q25, bccddmean_mean], 'color': '#dec7cb',
                                             'line': {'color': 'DarkSlateGray', 'width': 2}},
                                            {'range': [bccddmean_mean, bccddmean_q75],'color': '#dec7cb',
                                             'line': {'color': 'DarkSlateGray', 'width': 2}},
                                            {'range': [bccddmean_q75, bccddmean_max], 'color': '#de3a5b'}],
                                        'bar': {'color': 'black'}}))

                                fig_bccddmean.update_layout(height=200,
                                                            margin={'t':0, 'b':0, 'l':0})
                                
                                st.markdown(html_BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN, unsafe_allow_html=True)

                                # Go Indicator bullets
                                st.plotly_chart(fig_bccddmean) 
                                
                                # ==================== ViolinPlot ========================================================
                                sns.violinplot(x='PRED_CLASSE_CLIENT', y='BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN',
                                               data=df_dashboard,
                                               palette=['Green', 'Crimson'])

                                plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                         bccddmean_client,
                                         color="orange",
                                         marker="$\\bigotimes$", markersize=28)
                                plt.xlabel('TARGET', fontsize=16)
                                client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                       linestyle='None',
                                                       markersize=16, label='Position du client')
                                plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                st.set_option('deprecation.showPyplotGlobalUse', False)
                                st.pyplot()
                                    
                                # ==================== DistPlot ==========================================================
                                # Non-défaillants
                                sns.distplot(df_dashboard['BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN'][df_dashboard[
                                    'PRED_CLASSE_CLIENT'] == 0],
                                             label='Non-Défaillants', hist=False, color='Green')
                                # Défaillants
                                sns.distplot(df_dashboard['BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN'][df_dashboard[
                                    'PRED_CLASSE_CLIENT'] == 1],
                                             label='Défaillants', hist=False, color='Crimson')
                                plt.xlabel('BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN', fontsize=16)
                                plt.ylabel('Probability Density', fontsize=16)
                                plt.xticks(fontsize=16, rotation=90)
                                plt.yticks(fontsize=16)
                                # Position du client
                                plt.axvline(x=bccddmean_client, color='orange', label='Position du client')
                                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                st.set_option('deprecation.showPyplotGlobalUse', False)
                                st.pyplot()
                                
                                
                        # ==============================================================
                        # Variable BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN
                        # Moyenne de du ratio des prêts précédents sur d'autres institution de :
                        # la dette actuelle sur le crédit et la limite de crédit actuelle de la
                        # carte de crédit
                        # ==============================================================
                        if 'BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN...**'):
                                
                                bcdtcrm_min = int(df_client.BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN_MIN.values*100)
                                bcdtcrm_q25 = int(df_client.BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN_Q25.values*100)
                                bcdtcrm_mean = int(df_client.BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN_MEAN.values*100)
                                bcdtcrm_q75 = int(df_client.BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN_Q75.values*100)
                                bcdtcrm_max = int(df_client.BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN_MAX.values*100)
                                bcdtcrm_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                    'BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN'].values*100)
                                bcdtcrm_axis_min = min(bcdtcrm_min, bcdtcrm_client)
                                bcdtcrm_axis_max = max(bcdtcrm_max, bcdtcrm_client)
                                
                                cond = bcdtcrm_client == bcdtcrm_min and bcdtcrm_min == bcdtcrm_q25 \
                                    and bcdtcrm_q25 == bcdtcrm_mean and bcdtcrm_mean == bcdtcrm_q75 \
                                    and bcdtcrm_q75 == bcdtcrm_max
                                
                                if not cond:
                                                                                                    
                                    fig_bcdtcrm = go.Figure()
                                        
                                    fig_bcdtcrm.add_trace(go.Indicator(
                                       mode = "number+gauge+delta",
                                       value = bcdtcrm_client,
                                       delta = {'reference': bcdtcrm_mean,
                                                'increasing': {'color': 'Crimson'},
                                                'decreasing': {'color': 'Green'}},
                                       domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                       title = {'text': 'BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN',
                                                'font': {'size': 9}, 'align' : 'left'},
                                       gauge = {
                                           'shape': 'bullet',
                                           'axis': {'range': [bcdtcrm_axis_min, bcdtcrm_axis_max]},
                                           'threshold': {
                                               'line': {'color': 'black', 'width': 3},
                                               'thickness': 0.75,
                                               'value': bcdtcrm_client},
                                           'steps': [
                                               {'range': [0, bcdtcrm_min], 'color': 'white'},
                                               {'range': [bcdtcrm_min, bcdtcrm_q25], 'color': '#de3a5b'},
                                               {'range': [bcdtcrm_q25, bcdtcrm_mean], 'color': '#dec7cb',
                                                'line': {'color': 'DarkSlateGray', 'width': 2}},
                                               {'range': [bcdtcrm_mean, bcdtcrm_q75],'color': '#dec7cb',
                                                'line': {'color': 'DarkSlateGray', 'width': 2}},
                                               {'range': [bcdtcrm_q75, bcdtcrm_max], 'color': '#de3a5b'}],
                                           'bar': {'color': 'black'}}))
        
                                    fig_bcdtcrm.update_layout(height=200,
                                                               margin={'t':0, 'b':0, 'l':0})
                                    st.markdown(html_BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN, unsafe_allow_html=True)
        
                                    # Go Indicator bullets
                                    st.plotly_chart(fig_bcdtcrm) 
                                     
                                    # ==================== ViolinPlot ========================================================
                                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN',
                                                   data=df_dashboard,
                                                   palette=['Green', 'Crimson'])
    
                                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                             bcdtcrm_client,
                                             color="orange",
                                             marker="$\\bigotimes$", markersize=28)
                                    plt.xlabel('TARGET', fontsize=16)
                                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                           linestyle='None',
                                                           markersize=16, label='Position du client')
                                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()
                                        
                                    # ==================== DistPlot ==========================================================
                                    # Non-défaillants
                                    sns.distplot(df_dashboard['BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 0],
                                                 label='Non-Défaillants', hist=False, color='Green')
                                    # Défaillants
                                    sns.distplot(df_dashboard['BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 1],
                                                 label='Défaillants', hist=False, color='Crimson')
                                    plt.xlabel('BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN', fontsize=16)
                                    plt.ylabel('Probability Density', fontsize=16)
                                    plt.xticks(fontsize=16, rotation=90)
                                    plt.yticks(fontsize=16)
                                    # Position du client
                                    plt.axvline(x=bcdtcrm_client, color='orange', label='Position du client')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()
                                     
                                else:
                                    
                                    st.markdown(html_BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN, unsafe_allow_html=True)                                   
                                    st.write("Toutes les valeurs sont identiques")


                        # ==============================================================
                        # Variable CAR_EMPLOYED_RATIO
                        # Ratio : Âge de la voiture du demandeur / Ancienneté dans l'emploi à la
                        # date de la demande
                        # ==============================================================
                        if 'CAR_EMPLOYED_RATIO' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif CAR_EMPLOYED_RATIO...**'):
                                
                                cer_max = int(df_client.CAR_EMPLOYED_RATIO_MIN.values*1000)
                                cer_q75 = int(df_client.CAR_EMPLOYED_RATIO_Q25.values*1000)
                                cer_mean = int(df_client.CAR_EMPLOYED_RATIO_MEAN.values*1000)
                                cer_q25 = int(df_client.CAR_EMPLOYED_RATIO_Q75.values*1000)
                                cer_min = int(df_client.CAR_EMPLOYED_RATIO_MAX.values*1000)
                                cer_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                    'CAR_EMPLOYED_RATIO'].values*1000)
                                cer_axis_min = min(cer_min, cer_client)
                                cer_axis_max = max(cer_max, cer_client)
                                
                                cond = cer_client == cer_min and cer_min == cer_q25 and \
                                    cer_q25 == cer_mean and cer_mean == cer_q75 and \
                                    cer_q75 == cer_max
                                
                                if not cond:
                                    fig_cer = go.Figure()
                                
                                    fig_cer.add_trace(go.Indicator(
                                        mode = "number+gauge+delta",
                                        value = cer_client,
                                        delta = {'reference': cer_mean,
                                                 'increasing': {'color': 'Crimson'},
                                                 'decreasing': {'color': 'Green'}},
                                        domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                        title = {'text': 'CAR_EMPLOYED_RATIO',
                                                 'font': {'size': 9}, 'align' : 'left'},
                                        gauge = {
                                            'shape': 'bullet',
                                            'axis': {'range': [cer_axis_min, cer_axis_max]},
                                            'threshold': {
                                                'line': {'color': 'black', 'width': 3},
                                                'thickness': 0.75,
                                                'value': cer_client},
                                            'steps': [
                                                {'range': [0, cer_min], 'color': 'white'},
                                                {'range': [cer_min, cer_q25], 'color': '#de3a5b'},
                                                {'range': [cer_q25, cer_mean], 'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [cer_mean, cer_q75],'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [cer_q75, cer_max], 'color': '#de3a5b'}],
                                            'bar': {'color': 'black'}}))
                                
                                    fig_cer.update_layout(height=200,
                                                          margin={'t':0, 'b':0, 'l':0})
  
                                    st.markdown(html_CAR_EMPLOYED_RATIO, unsafe_allow_html=True)
        
                                    # Go Indicator bullets
                                    st.plotly_chart(fig_cer) 

                                    # ==================== ViolinPlot ========================================================
                                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='CAR_EMPLOYED_RATIO',
                                                   data=df_dashboard,
                                                   palette=['Green', 'Crimson'])
    
                                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                             cer_client,
                                             color="orange",
                                             marker="$\\bigotimes$", markersize=28)
                                    plt.xlabel('TARGET', fontsize=16)
                                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                           linestyle='None',
                                                           markersize=16, label='Position du client')
                                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()
                                        
                                    # ==================== DistPlot ==========================================================
                                    # Non-défaillants
                                    sns.distplot(df_dashboard['CAR_EMPLOYED_RATIO'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 0],
                                                 label='Non-Défaillants', hist=False, color='Green')
                                    # Défaillants
                                    sns.distplot(df_dashboard['CAR_EMPLOYED_RATIO'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 1],
                                                 label='Défaillants', hist=False, color='Crimson')
                                    plt.xlabel('CAR_EMPLOYED_RATIO', fontsize=16)
                                    plt.ylabel('Probability Density', fontsize=16)
                                    plt.xticks(fontsize=16, rotation=90)
                                    plt.yticks(fontsize=16)
                                    # Position du client
                                    plt.axvline(x=cer_client, color='orange', label='Position du client')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()
                                    
                                else:
                                    
                                    st.markdown(html_CAR_EMPLOYED_RATIO, unsafe_allow_html=True)                                   
                                    st.write("Toutes les valeurs sont identiques")                                   


                        # ==============================================================
                        # Variable CREDIT_ANNUITY_RATIO
                        # Ratio : montant du crédit du prêt / Annuité de prêt 
                        # ==============================================================
                        if 'CREDIT_ANNUITY_RATIO' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif CREDIT_ANNUITY_RATIO...**'):
                                
                                car_min = int(df_client.CREDIT_ANNUITY_RATIO_MIN.values)
                                car_q25 = int(df_client.CREDIT_ANNUITY_RATIO_Q25.values)
                                car_mean = int(df_client.CREDIT_ANNUITY_RATIO_MEAN.values)
                                car_q75 = int(df_client.CREDIT_ANNUITY_RATIO_Q75.values)
                                car_max = int(df_client.CREDIT_ANNUITY_RATIO_MAX.values)
                                car_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                    'CREDIT_ANNUITY_RATIO'].values)
                                car_axis_min = min(car_min, car_client)
                                car_axis_max = max(car_max, car_client)                                
                                
                                cond = car_client == car_min and car_min == car_q25 and \
                                    car_q25 == car_mean and car_mean == car_q75 \
                                    and car_q75 == car_max
                                
                                if not cond:
                                    fig_car = go.Figure()
                                
                                    fig_car.add_trace(go.Indicator(
                                        mode = "number+gauge+delta",
                                        value = car_client,
                                        delta = {'reference': car_mean,
                                                 'increasing': {'color': 'Crimson'},
                                                 'decreasing': {'color': 'Green'}},
                                        domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                        title = {'text': 'CREDIT_ANNUITY_RATIO',
                                                 'font': {'size': 9}, 'align' : 'left'},
                                        gauge = {
                                            'shape': 'bullet',
                                            'axis': {'range': [car_axis_min, car_axis_max]},
                                            'threshold': {
                                                'line': {'color': 'black', 'width': 3},
                                                'thickness': 0.75,
                                                'value': car_client},
                                            'steps': [
                                                {'range': [0, car_min], 'color': 'white'},
                                                {'range': [car_min, car_q25], 'color': '#de3a5b'},
                                                {'range': [car_q25, car_mean], 'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [car_mean, car_q75],'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [car_q75, car_max], 'color': '#de3a5b'}],
                                            'bar': {'color': 'black'}}))

                                    fig_car.update_layout(height=200,
                                                          margin={'t':0, 'b':0, 'l':0})
  
                                    st.markdown(html_CREDIT_ANNUITY_RATIO, unsafe_allow_html=True)
        
                                    # Go Indicator bullets
                                    st.plotly_chart(fig_car) 
                                    
                                    # ==================== ViolinPlot ========================================================
                                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='CREDIT_ANNUITY_RATIO',
                                                   data=df_dashboard,
                                                   palette=['Green', 'Crimson'])
    
                                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                             car_client,
                                             color="orange",
                                             marker="$\\bigotimes$", markersize=28)
                                    plt.xlabel('TARGET', fontsize=16)
                                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                           linestyle='None',
                                                           markersize=16, label='Position du client')
                                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()
                                        
                                    # ==================== DistPlot ==========================================================
                                    # Non-défaillants
                                    sns.distplot(df_dashboard['CREDIT_ANNUITY_RATIO'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 0],
                                                 label='Non-Défaillants', hist=False, color='Green')
                                    # Défaillants
                                    sns.distplot(df_dashboard['CREDIT_ANNUITY_RATIO'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 1],
                                                 label='Défaillants', hist=False, color='Crimson')
                                    plt.xlabel('CREDIT_ANNUITY_RATIO', fontsize=16)
                                    plt.ylabel('Probability Density', fontsize=16)
                                    plt.xticks(fontsize=16, rotation=90)
                                    plt.yticks(fontsize=16)
                                    # Position du client
                                    plt.axvline(x=car_client, color='orange', label='Position du client')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()
                                    
                                else:
                                    
                                    st.markdown(html_CREDIT_ANNUITY_RATIO, unsafe_allow_html=True)                                   
                                    st.write("Toutes les valeurs sont identiques")                                   


                        # ==============================================================
                        # Variable CREDIT_GOODS_RATIO
                        # Ratio : Montant du crédit du prêt / prix des biens pour lesquels le prêt
                        # est accordé / Crédit est supérieur au prix des biens ?  
                        # ==============================================================
                        if 'CREDIT_GOODS_RATIO' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif CREDIT_GOODS_RATIO...**'):
                                
                                cgr_min = int(df_client.CREDIT_GOODS_RATIO_MIN.values * 100)
                                cgr_q25 = int(df_client.CREDIT_GOODS_RATIO_Q25.values * 100)
                                cgr_mean = int(df_client.CREDIT_GOODS_RATIO_MEAN.values * 100)
                                cgr_q75 = int(df_client.CREDIT_GOODS_RATIO_Q75.values * 100)
                                cgr_max = int(df_client.CREDIT_GOODS_RATIO_MAX.values * 100)
                                cgr_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                    'CREDIT_GOODS_RATIO'].values * 100)
                                cgr_axis_min = min(cgr_min, cgr_client)
                                cgr_axis_max = max(cgr_max, cgr_client)                                
                                
                                cond = cgr_client == cgr_min and cgr_min == cgr_q25 \
                                    and cgr_q25 == cgr_mean and cgr_mean == cgr_q75 \
                                    and cgr_q75 == cgr_max
                                
                                if not cond:
                                    fig_cgr = go.Figure()
                                
                                    fig_cgr.add_trace(go.Indicator(
                                        mode = "number+gauge+delta",
                                        value = cgr_client,
                                        delta = {'reference': cgr_mean,
                                                 'increasing': {'color': 'Crimson'},
                                                 'decreasing': {'color': 'Green'}},
                                        domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                        title = {'text': 'CREDIT_GOODS_RATIO',
                                                 'font': {'size': 9}, 'align' : 'left'},
                                        gauge = {
                                            'shape': 'bullet',
                                            'axis': {'range': [cgr_axis_min, cgr_axis_max]},
                                            'threshold': {
                                                'line': {'color': 'black', 'width': 3},
                                                'thickness': 0.75,
                                                'value': cgr_client},
                                            'steps': [
                                                {'range': [0, cgr_min], 'color': 'white'},
                                                {'range': [cgr_min, cgr_q25], 'color': '#de3a5b'},
                                                {'range': [cgr_q25, cgr_mean], 'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [cgr_mean, cgr_q75],'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [cgr_q75, cgr_max], 'color': '#de3a5b'}],
                                            'bar': {'color': 'black'}}))

                                    fig_cgr.update_layout(height=200,
                                                          margin={'t':0, 'b':0, 'l':0})
  
                                    st.markdown(html_CREDIT_GOODS_RATIO, unsafe_allow_html=True)
        
                                    # Go Indicator bullets
                                    st.plotly_chart(fig_cgr) 
                                    
                                    # ==================== ViolinPlot ========================================================
                                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='CREDIT_GOODS_RATIO',
                                                   data=df_dashboard,
                                                   palette=['Green', 'Crimson'])
    
                                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                             cgr_client,
                                             color="orange",
                                             marker="$\\bigotimes$", markersize=28)
                                    plt.xlabel('TARGET', fontsize=16)
                                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                           linestyle='None',
                                                           markersize=16, label='Position du client')
                                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()
                                        
                                    # ==================== DistPlot ==========================================================
                                    # Non-défaillants
                                    sns.distplot(df_dashboard['CREDIT_GOODS_RATIO'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 0],
                                                 label='Non-Défaillants', hist=False, color='Green')
                                    # Défaillants
                                    sns.distplot(df_dashboard['CREDIT_GOODS_RATIO'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 1],
                                                 label='Défaillants', hist=False, color='Crimson')
                                    plt.xlabel('CREDIT_GOODS_RATIO', fontsize=16)
                                    plt.ylabel('Probability Density', fontsize=16)
                                    plt.xticks(fontsize=16, rotation=90)
                                    plt.yticks(fontsize=16)
                                    # Position du client
                                    plt.axvline(x=cgr_client, color='orange', label='Position du client')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()                                   
                                    
                                else:
                                    
                                    st.markdown(html_CREDIT_GOODS_RATIO, unsafe_allow_html=True)                                   
                                    st.write("Toutes les valeurs sont identiques") 


                        # ==============================================================
                        # Variable YEAR_BIRTH
                        # Âge (ans) 
                        # ==============================================================
                        if 'YEAR_BIRTH' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif YEAR_BIRTH...**'):
                                
                                age_max = int(-df_client.DAYS_BIRTH_MIN.values/365)
                                age_q75 = int(-df_client.DAYS_BIRTH_Q25.values/365)
                                age_mean = int(-df_client.DAYS_BIRTH_MEAN.values/365)
                                age_q25 = int(-df_client.DAYS_BIRTH_Q75.values/365)
                                age_min = int(-df_client.DAYS_BIRTH_MAX.values/365)
                                age_client = int(-df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                    'DAYS_BIRTH'].values/365)
                                age_axis_min = min(age_min, age_client)
                                age_axis_max = max(age_max, age_client)   
                                
                                cond = age_client == age_min and age_min == age_q25 and \
                                    age_q25 == age_mean and age_mean == age_q75 and age_q75 == age_max
                                
                                if not cond:
                                    fig_age = go.Figure()
                                
                                    fig_age.add_trace(go.Indicator(
                                        mode = "number+gauge+delta",
                                        value = age_client,
                                        delta = {'reference': age_mean,
                                                 'increasing': {'color': 'Crimson'},
                                                 'decreasing': {'color': 'Green'}},
                                        domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                        title = {'text': 'Âge',
                                                 'font': {'size': 9}, 'align' : 'left'},
                                        gauge = {
                                            'shape': 'bullet',
                                            'axis': {'range': [age_axis_min, age_axis_max]},
                                            'threshold': {
                                                'line': {'color': 'black', 'width': 3},
                                                'thickness': 0.75,
                                                'value': age_client},
                                            'steps': [
                                                {'range': [0, age_min], 'color': 'white'},
                                                {'range': [age_min, age_q25], 'color': '#de3a5b'},
                                                {'range': [age_q25, age_mean], 'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [age_mean, age_q75],'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [age_q75, age_max], 'color': '#de3a5b'}],
                                            'bar': {'color': 'black'}}))

                                    fig_age.update_layout(height=200,
                                                           margin={'t':0, 'b':0, 'l':0})
  
                                    st.markdown(html_YEAR_BIRTH, unsafe_allow_html=True)
        
                                    # Go Indicator bullets
                                    st.plotly_chart(fig_age) 
                                    
                                    # ==================== ViolinPlot ========================================================
                                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='YEAR_BIRTH',
                                                   data=df_dashboard,
                                                   palette=['Green', 'Crimson'])
    
                                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                             age_client,
                                             color="orange",
                                             marker="$\\bigotimes$", markersize=28)
                                    plt.xlabel('TARGET', fontsize=16)
                                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                           linestyle='None',
                                                           markersize=16, label='Position du client')
                                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()
                                        
                                    # ==================== DistPlot ==========================================================
                                    # Non-défaillants
                                    sns.distplot(df_dashboard['YEAR_BIRTH'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 0],
                                                 label='Non-Défaillants', hist=False, color='Green')
                                    # Défaillants
                                    sns.distplot(df_dashboard['YEAR_BIRTH'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 1],
                                                 label='Défaillants', hist=False, color='Crimson')
                                    plt.xlabel('YEAR_BIRTH', fontsize=16)
                                    plt.ylabel('Probability Density', fontsize=16)
                                    plt.xticks(fontsize=16, rotation=90)
                                    plt.yticks(fontsize=16)
                                    # Position du client
                                    plt.axvline(x=age_client, color='orange', label='Position du client')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()                                   
                                                                        
                                else:
                                    
                                    st.markdown(html_YEAR_BIRTH, unsafe_allow_html=True)                                   
                                    st.write("Toutes les valeurs sont identiques") 


                        # ==============================================================
                        # Variable YEAR_ID_PUBLISH
                        # Combien de jours avant la demande le client a-t-il changé la pièce
                        # d'identité avec laquelle il a demandé le prêt ? (ans)
                        # ==============================================================
                        if 'YEAR_ID_PUBLISH' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif YEAR_ID_PUBLISH...**'):
                                
                                dip_max = int(-df_client.DAYS_ID_PUBLISH_MIN.values/365)
                                dip_q75 = int(-df_client.DAYS_ID_PUBLISH_Q25.values/365)
                                dip_mean = int(-df_client.DAYS_ID_PUBLISH_MEAN.values/365)
                                dip_q25 = int(-df_client.DAYS_ID_PUBLISH_Q75.values/365)
                                dip_min = int(-df_client.DAYS_ID_PUBLISH_MAX.values/365)
                                dip_client = int(-df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                    'DAYS_ID_PUBLISH'].values/365)
                                dip_axis_min = min(dip_min, dip_client)
                                dip_axis_max = max(dip_max, dip_client) 
                                
                                cond = dip_client == dip_min and dip_min == dip_q25 and \
                                    dip_q25 == dip_mean and dip_mean == dip_q75 and dip_q75 == dip_max
                                
                                if not cond:
                                    fig_dip = go.Figure()
                                
                                    fig_dip.add_trace(go.Indicator(
                                        mode = "number+gauge+delta",
                                        value = dip_client,
                                        delta = {'reference': dip_mean,
                                                 'increasing': {'color': 'Crimson'},
                                                 'decreasing': {'color': 'Green'}},
                                        domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                        title = {'text': 'DAYS_ID_PUBLISH',
                                                 'font': {'size': 9}, 'align' : 'left'},
                                        gauge = {
                                            'shape': 'bullet',
                                            'axis': {'range': [dip_axis_min, dip_axis_max]},
                                            'threshold': {
                                                'line': {'color': 'black', 'width': 3},
                                                'thickness': 0.75,
                                                'value': dip_client},
                                            'steps': [
                                                {'range': [0, dip_min], 'color': 'white'},
                                                {'range': [dip_min, dip_q25], 'color': '#de3a5b'},
                                                {'range': [dip_q25, dip_mean], 'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [dip_mean, dip_q75],'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [dip_q75, dip_max], 'color': '#de3a5b'}],
                                            'bar': {'color': 'black'}}))

                                    fig_dip.update_layout(height=200,
                                                          margin={'t':0, 'b':0, 'l':0})
  
                                    st.markdown(html_YEAR_ID_PUBLISH, unsafe_allow_html=True)
        
                                    # Go Indicator bullets
                                    st.plotly_chart(fig_dip) 
                                     
                                    # ==================== ViolinPlot ========================================================
                                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='DAYS_ID_PUBLISH',
                                                   data=df_dashboard,
                                                   palette=['Green', 'Crimson'])
    
                                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                             dip_client,
                                             color="orange",
                                             marker="$\\bigotimes$", markersize=28)
                                    plt.xlabel('TARGET', fontsize=16)
                                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                           linestyle='None',
                                                           markersize=16, label='Position du client')
                                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()
                                        
                                    # ==================== DistPlot ==========================================================
                                    # Non-défaillants
                                    sns.distplot(df_dashboard['DAYS_ID_PUBLISH'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 0],
                                                 label='Non-Défaillants', hist=False, color='Green')
                                    # Défaillants
                                    sns.distplot(df_dashboard['DAYS_ID_PUBLISH'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 1],
                                                 label='Défaillants', hist=False, color='Crimson')
                                    plt.xlabel('DAYS_ID_PUBLISH', fontsize=16)
                                    plt.ylabel('Probability Density', fontsize=16)
                                    plt.xticks(fontsize=16, rotation=90)
                                    plt.yticks(fontsize=16)
                                    # Position du client
                                    plt.axvline(x=dip_client, color='orange', label='Position du client')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()                                   
                                                                                                           
                                else:
                                    
                                    st.markdown(html_YEAR_ID_PUBLISH, unsafe_allow_html=True)                                   
                                    st.write("Toutes les valeurs sont identiques") 


                        # ==============================================================
                        # Variable EXT_SOURCE_1
                        # Source externe normalisée 
                        # ==============================================================
                        if 'EXT_SOURCE_1' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif EXT_SOURCE_1...**'):
 
                                es1_min = int(df_client.EXT_SOURCE_1_MIN.values * 100)
                                es1_q25 = int(df_client.EXT_SOURCE_1_Q25.values * 100)
                                es1_mean = int(df_client.EXT_SOURCE_1_MEAN.values * 100)
                                es1_q75 = int(df_client.EXT_SOURCE_1_Q75.values * 100)
                                es1_max = int(df_client.EXT_SOURCE_1_MAX.values * 100)
                                es1_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                    'EXT_SOURCE_1'].values * 100)
                                es1_axis_min = min(es1_min, es1_client)
                                es1_axis_max = max(es1_max, es1_client) 
                                
                                cond = es1_client == es1_min and es1_min == es1_q25 and \
                                    es1_q25 == es1_mean and es1_mean == es1_q75 and \
                                    es1_q75 == es1_max
                                
                                if not cond:
                                    fig_es1 = go.Figure()
                                
                                    fig_es1.add_trace(go.Indicator(
                                        mode = "number+gauge+delta",
                                        value = es1_client,
                                        delta = {'reference': es1_mean,
                                                 'increasing': {'color': 'Crimson'},
                                                 'decreasing': {'color': 'Green'}},
                                        domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                        title = {'text': 'EXT_SOURCE_1',
                                                 'font': {'size': 9}, 'align' : 'left'},
                                        gauge = {
                                            'shape': 'bullet',
                                            'axis': {'range': [es1_axis_min, es1_axis_max]},
                                            'threshold': {
                                                'line': {'color': 'black', 'width': 3},
                                                'thickness': 0.75,
                                                'value': es1_client},
                                            'steps': [
                                                {'range': [0, es1_min], 'color': 'white'},
                                                {'range': [es1_min, es1_q25], 'color': '#de3a5b'},
                                                {'range': [es1_q25, es1_mean], 'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [es1_mean, es1_q75],'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [es1_q75, es1_max], 'color': '#de3a5b'}],
                                            'bar': {'color': 'black'}}))

                                    fig_es1.update_layout(height=200,
                                                          margin={'t':0, 'b':0, 'l':0})
  
                                    st.markdown(html_EXT_SOURCE_1, unsafe_allow_html=True)
        
                                    # Go Indicator bullets
                                    st.plotly_chart(fig_es1) 
                                     
                                    # ==================== ViolinPlot ========================================================
                                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='EXT_SOURCE_1',
                                                   data=df_dashboard,
                                                   palette=['Green', 'Crimson'])
    
                                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                             es1_client,
                                             color="orange",
                                             marker="$\\bigotimes$", markersize=28)
                                    plt.xlabel('TARGET', fontsize=16)
                                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                           linestyle='None',
                                                           markersize=16, label='Position du client')
                                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()
                                        
                                    # ==================== DistPlot ==========================================================
                                    # Non-défaillants
                                    sns.distplot(df_dashboard['EXT_SOURCE_1'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 0],
                                                 label='Non-Défaillants', hist=False, color='Green')
                                    # Défaillants
                                    sns.distplot(df_dashboard['EXT_SOURCE_1'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 1],
                                                 label='Défaillants', hist=False, color='Crimson')
                                    plt.xlabel('EXT_SOURCE_1', fontsize=16)
                                    plt.ylabel('Probability Density', fontsize=16)
                                    plt.xticks(fontsize=16, rotation=90)
                                    plt.yticks(fontsize=16)
                                    # Position du client
                                    plt.axvline(x=es1_client, color='orange', label='Position du client')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()                                   
                                                                                                           
                                else:
                                    
                                    st.markdown(html_EXT_SOURCE_1, unsafe_allow_html=True)                                   
                                    st.write("Toutes les valeurs sont identiques") 


                        # ==============================================================
                        # Variable EXT_SOURCE_2
                        # Source externe normalisée 
                        # ==============================================================
                        if 'EXT_SOURCE_2' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif EXT_SOURCE_2...**'):
 
                                es2_min = int(df_client.EXT_SOURCE_2_MIN.values * 100)
                                es2_q25 = int(df_client.EXT_SOURCE_2_Q25.values * 100)
                                es2_mean = int(df_client.EXT_SOURCE_2_MEAN.values * 100)
                                es2_q75 = int(df_client.EXT_SOURCE_2_Q75.values * 100)
                                es2_max = int(df_client.EXT_SOURCE_2_MAX.values * 100)
                                es2_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                    'EXT_SOURCE_2'].values * 100)
                                es2_axis_min = min(es2_min, es2_client)
                                es2_axis_max = max(es2_max, es2_client) 
                                    
                                cond = es2_client == es2_min and es2_min == es2_q25 and \
                                    es2_q25 == es2_mean and es2_mean == es2_q75 and \
                                    es2_q75 == es2_max
                                
                                if not cond:
                                    fig_es2 = go.Figure()
                                
                                    fig_es2.add_trace(go.Indicator(
                                        mode = "number+gauge+delta",
                                        value = es2_client,
                                        delta = {'reference': es2_mean,
                                                 'increasing': {'color': 'Crimson'},
                                                 'decreasing': {'color': 'Green'}},
                                        domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                        title = {'text': 'EXT_SOURCE_2',
                                                 'font': {'size': 9}, 'align' : 'left'},
                                        gauge = {
                                            'shape': 'bullet',
                                            'axis': {'range': [es2_axis_min, es2_axis_max]},
                                            'threshold': {
                                                'line': {'color': 'black', 'width': 3},
                                                'thickness': 0.75,
                                                'value': es2_client},
                                            'steps': [
                                                {'range': [0, es2_min], 'color': 'white'},
                                                {'range': [es2_min, es2_q25], 'color': '#de3a5b'},
                                                {'range': [es2_q25, es2_mean], 'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [es2_mean, es2_q75],'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [es2_q75, es2_max], 'color': '#de3a5b'}],
                                            'bar': {'color': 'black'}}))

                                    fig_es2.update_layout(height=200,
                                                          margin={'t':0, 'b':0, 'l':0})
  
                                    st.markdown(html_EXT_SOURCE_2, unsafe_allow_html=True)
        
                                    # Go Indicator bullets
                                    st.plotly_chart(fig_es2) 

                                    # ==================== ViolinPlot ========================================================
                                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='EXT_SOURCE_2',
                                                   data=df_dashboard,
                                                   palette=['Green', 'Crimson'])
    
                                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                             es2_client,
                                             color="orange",
                                             marker="$\\bigotimes$", markersize=28)
                                    plt.xlabel('TARGET', fontsize=16)
                                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                           linestyle='None',
                                                           markersize=16, label='Position du client')
                                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()
                                        
                                    # ==================== DistPlot ==========================================================
                                    # Non-défaillants
                                    sns.distplot(df_dashboard['EXT_SOURCE_2'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 0],
                                                 label='Non-Défaillants', hist=False, color='Green')
                                    # Défaillants
                                    sns.distplot(df_dashboard['EXT_SOURCE_2'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 1],
                                                 label='Défaillants', hist=False, color='Crimson')
                                    plt.xlabel('EXT_SOURCE_2', fontsize=16)
                                    plt.ylabel('Probability Density', fontsize=16)
                                    plt.xticks(fontsize=16, rotation=90)
                                    plt.yticks(fontsize=16)
                                    # Position du client
                                    plt.axvline(x=es2_client, color='orange', label='Position du client')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()                                   
                                   
                                else:
                                    
                                    st.markdown(html_EXT_SOURCE_2, unsafe_allow_html=True)                                   
                                    st.write("Toutes les valeurs sont identiques") 


                        # ==============================================================
                        # Variable EXT_SOURCE_3
                        # Source externe normalisée 
                        # ==============================================================
                        if 'EXT_SOURCE_3' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif EXT_SOURCE_3...**'):
 
                                es3_min = int(df_client.EXT_SOURCE_3_MIN.values * 100)
                                es3_q25 = int(df_client.EXT_SOURCE_3_Q25.values * 100)
                                es3_mean = int(df_client.EXT_SOURCE_3_MEAN.values * 100)
                                es3_q75 = int(df_client.EXT_SOURCE_3_Q75.values * 100)
                                es3_max = int(df_client.EXT_SOURCE_3_MAX.values * 100)
                                es3_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                    'EXT_SOURCE_3'].values * 100)
                                es3_axis_min = min(es3_min, es3_client)
                                es3_axis_max = max(es3_max, es3_client) 
                                
                                cond = es3_client == es3_min and es3_min == es3_q25 and \
                                    es3_q25 == es3_mean and es3_mean == es3_q75 and \
                                    es3_q75 == es3_max
                                
                                if not cond:
                                    fig_es3 = go.Figure()
                                
                                    fig_es3.add_trace(go.Indicator(
                                        mode = "number+gauge+delta",
                                        value = es3_client,
                                        delta = {'reference': es3_mean,
                                                 'increasing': {'color': 'Crimson'},
                                                 'decreasing': {'color': 'Green'}},
                                        domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                        title = {'text': 'EXT_SOURCE_3',
                                                 'font': {'size': 9}, 'align' : 'left'},
                                        gauge = {
                                            'shape': 'bullet',
                                            'axis': {'range': [es3_axis_min, es3_axis_max]},
                                            'threshold': {
                                                'line': {'color': 'black', 'width': 3},
                                                'thickness': 0.75,
                                                'value': es3_client},
                                            'steps': [
                                                {'range': [0, es3_min], 'color': 'white'},
                                                {'range': [es3_min, es3_q25], 'color': '#de3a5b'},
                                                {'range': [es3_q25, es3_mean], 'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [es3_mean, es3_q75],'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [es3_q75, es3_max], 'color': '#de3a5b'}],
                                            'bar': {'color': 'black'}}))

                                    fig_es3.update_layout(height=200,
                                                          margin={'t':0, 'b':0, 'l':0})
  
                                    st.markdown(html_EXT_SOURCE_3, unsafe_allow_html=True)
        
                                    # Go Indicator bullets
                                    st.plotly_chart(fig_es3) 

                                    # ==================== ViolinPlot ========================================================
                                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='EXT_SOURCE_3',
                                                   data=df_dashboard,
                                                   palette=['Green', 'Crimson'])
    
                                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                             es3_client,
                                             color="orange",
                                             marker="$\\bigotimes$", markersize=28)
                                    plt.xlabel('TARGET', fontsize=16)
                                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                           linestyle='None',
                                                           markersize=16, label='Position du client')
                                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()
                                        
                                    # ==================== DistPlot ==========================================================
                                    # Non-défaillants
                                    sns.distplot(df_dashboard['EXT_SOURCE_3'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 0],
                                                 label='Non-Défaillants', hist=False, color='Green')
                                    # Défaillants
                                    sns.distplot(df_dashboard['EXT_SOURCE_3'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 1],
                                                 label='Défaillants', hist=False, color='Crimson')
                                    plt.xlabel('EXT_SOURCE_3', fontsize=16)
                                    plt.ylabel('Probability Density', fontsize=16)
                                    plt.xticks(fontsize=16, rotation=90)
                                    plt.yticks(fontsize=16)
                                    # Position du client
                                    plt.axvline(x=es3_client, color='orange', label='Position du client')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()                                   
                                    
                                else:
                                    
                                    st.markdown(html_EXT_SOURCE_3, unsafe_allow_html=True)                                   
                                    st.write("Toutes les valeurs sont identiques") 


                        # ==============================================================
                        # Variable EXT_SOURCE_MAX
                        # Valeur maximale des 3 sources externes normalisées (EXT_SOURCE_1, EXT_SOURCE_2 et EXT_SOURCE_3)
                        # ==============================================================
                        if 'EXT_SOURCE_MAX' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif EXT_SOURCE_MAX...**'):
 
                                esm_min = int(df_client.EXT_SOURCE_MAX_MIN.values * 100)
                                esm_q25 = int(df_client.EXT_SOURCE_MAX_Q25.values * 100)
                                esm_mean = int(df_client.EXT_SOURCE_MAX_MEAN.values * 100)
                                esm_q75 = int(df_client.EXT_SOURCE_MAX_Q75.values * 100)
                                esm_max = int(df_client.EXT_SOURCE_MAX_MAX.values * 100)
                                esm_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                    'EXT_SOURCE_MAX'].values * 100)
                                esm_axis_min = min(esm_min, esm_client)
                                esm_axis_max = max(esm_max, esm_client) 
                                
                                cond = esm_client == esm_min and esm_min == esm_q25 and \
                                    esm_q25 == esm_mean and esm_mean == esm_q75 and \
                                    esm_q75 == esm_max
                                
                                if not cond:
                                    fig_esm = go.Figure()
                                
                                    fig_esm.add_trace(go.Indicator(
                                        mode = "number+gauge+delta",
                                        value = esm_client,
                                        delta = {'reference': esm_mean,
                                                 'increasing': {'color': 'Crimson'},
                                                 'decreasing': {'color': 'Green'}},
                                        domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                        title = {'text': 'EXT_SOURCE_MAX',
                                                 'font': {'size': 9}, 'align' : 'left'},
                                        gauge = {
                                            'shape': 'bullet',
                                            'axis': {'range': [esm_axis_min, esm_axis_max]},
                                            'threshold': {
                                                'line': {'color': 'black', 'width': 3},
                                                'thickness': 0.75,
                                                'value': esm_client},
                                            'steps': [
                                                {'range': [0, esm_min], 'color': 'white'},
                                                {'range': [esm_min, esm_q25], 'color': '#de3a5b'},
                                                {'range': [esm_q25, esm_mean], 'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [esm_mean, esm_q75],'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [esm_q75, esm_max], 'color': '#de3a5b'}],
                                            'bar': {'color': 'black'}}))

                                    fig_esm.update_layout(height=200,
                                                          margin={'t':0, 'b':0, 'l':0})
  
                                    st.markdown(html_EXT_SOURCE_MAX, unsafe_allow_html=True)
        
                                    # Go Indicator bullets
                                    st.plotly_chart(fig_esm) 

                                    # ==================== ViolinPlot ========================================================
                                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='EXT_SOURCE_MAX',
                                                   data=df_dashboard,
                                                   palette=['Green', 'Crimson'])
    
                                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                             esm_client,
                                             color="orange",
                                             marker="$\\bigotimes$", markersize=28)
                                    plt.xlabel('TARGET', fontsize=16)
                                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                           linestyle='None',
                                                           markersize=16, label='Position du client')
                                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()
                                        
                                    # ==================== DistPlot ==========================================================
                                    # Non-défaillants
                                    sns.distplot(df_dashboard['EXT_SOURCE_MAX'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 0],
                                                 label='Non-Défaillants', hist=False, color='Green')
                                    # Défaillants
                                    sns.distplot(df_dashboard['EXT_SOURCE_MAX'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 1],
                                                 label='Défaillants', hist=False, color='Crimson')
                                    plt.xlabel('EXT_SOURCE_MAX', fontsize=16)
                                    plt.ylabel('Probability Density', fontsize=16)
                                    plt.xticks(fontsize=16, rotation=90)
                                    plt.yticks(fontsize=16)
                                    # Position du client
                                    plt.axvline(x=esm_client, color='orange', label='Position du client')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()                                   
                                                                        
                                else:
                                    
                                    st.markdown(html_EXT_SOURCE_MAX, unsafe_allow_html=True)                                   
                                    st.write("Toutes les valeurs sont identiques") 


                        # ==============================================================
                        # Variable EXT_SOURCE_SUM
                        # Somme des 3 sources externes normalisées (EXT_SOURCE_1, EXT_SOURCE_2 et EXT_SOURCE_3) 
                        # ==============================================================
                        if 'EXT_SOURCE_SUM' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif EXT_SOURCE_SUM...**'):
                                 
                                ess_min = int(df_client.EXT_SOURCE_SUM_MIN.values * 100)
                                ess_q25 = int(df_client.EXT_SOURCE_SUM_Q25.values * 100)
                                ess_mean = int(df_client.EXT_SOURCE_SUM_MEAN.values * 100)
                                ess_q75 = int(df_client.EXT_SOURCE_SUM_Q75.values * 100)
                                ess_max = int(df_client.EXT_SOURCE_SUM_MAX.values * 100)
                                ess_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                    'EXT_SOURCE_SUM'].values * 100)
                                ess_axis_min = min(ess_min, ess_client)
                                ess_axis_max = max(ess_max, ess_client) 
                                
                                cond = ess_client == ess_min and ess_min == ess_q25 and \
                                    ess_q25 == ess_mean and ess_mean == ess_q75 and \
                                    ess_q75 == ess_max
                                
                                if not cond:
                                    fig_ess = go.Figure()
                                
                                    fig_ess.add_trace(go.Indicator(
                                        mode = "number+gauge+delta",
                                        value = ess_client,
                                        delta = {'reference': ess_mean,
                                                 'increasing': {'color': 'Crimson'},
                                                 'decreasing': {'color': 'Green'}},
                                        domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                        title = {'text': 'EXT_SOURCE_SUM',
                                                 'font': {'size': 9}, 'align' : 'left'},
                                        gauge = {
                                            'shape': 'bullet',
                                            'axis': {'range': [ess_axis_min, ess_axis_max]},
                                            'threshold': {
                                                'line': {'color': 'black', 'width': 3},
                                                'thickness': 0.75,
                                                'value': ess_client},
                                            'steps': [
                                                {'range': [0, ess_min], 'color': 'white'},
                                                {'range': [ess_min, ess_q25], 'color': '#de3a5b'},
                                                {'range': [ess_q25, ess_mean], 'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [ess_mean, ess_q75],'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [ess_q75, ess_max], 'color': '#de3a5b'}],
                                            'bar': {'color': 'black'}}))

                                    fig_ess.update_layout(height=200,
                                                          margin={'t':0, 'b':0, 'l':0})
  
                                    st.markdown(html_EXT_SOURCE_SUM, unsafe_allow_html=True)
        
                                    # Go Indicator bullets
                                    st.plotly_chart(fig_ess) 

                                    # ==================== ViolinPlot ========================================================
                                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='EXT_SOURCE_SUM',
                                                   data=df_dashboard,
                                                   palette=['Green', 'Crimson'])
    
                                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                             ess_client,
                                             color="orange",
                                             marker="$\\bigotimes$", markersize=28)
                                    plt.xlabel('TARGET', fontsize=16)
                                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                           linestyle='None',
                                                           markersize=16, label='Position du client')
                                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()
                                        
                                    # ==================== DistPlot ==========================================================
                                    # Non-défaillants
                                    sns.distplot(df_dashboard['EXT_SOURCE_SUM'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 0],
                                                 label='Non-Défaillants', hist=False, color='Green')
                                    # Défaillants
                                    sns.distplot(df_dashboard['EXT_SOURCE_SUM'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 1],
                                                 label='Défaillants', hist=False, color='Crimson')
                                    plt.xlabel('EXT_SOURCE_SUM', fontsize=16)
                                    plt.ylabel('Probability Density', fontsize=16)
                                    plt.xticks(fontsize=16, rotation=90)
                                    plt.yticks(fontsize=16)
                                    # Position du client
                                    plt.axvline(x=ess_client, color='orange', label='Position du client')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()                                   
                                    
                                else:
                                    
                                    st.markdown(html_EXT_SOURCE_SUM, unsafe_allow_html=True)                                   
                                    st.write("Toutes les valeurs sont identiques") 


                        # ==============================================================
                        # Variable INST_PAY_AMT_INSTALMENT_SUM
                        # Somme du montant de l'acompte prescrit des crédits précédents sur cet
                        # acompte
                        # ==============================================================
                        if 'INST_PAY_AMT_INSTALMENT_SUM' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif INST_PAY_AMT_INSTALMENT_SUM...**'):
                                 
                                ipais_min = int(df_client.INST_PAY_AMT_INSTALMENT_SUM_MIN.values)
                                ipais_q25 = int(df_client.INST_PAY_AMT_INSTALMENT_SUM_Q25.values)
                                ipais_mean = int(df_client.INST_PAY_AMT_INSTALMENT_SUM_MEAN.values)
                                ipais_q75 = int(df_client.INST_PAY_AMT_INSTALMENT_SUM_Q75.values)
                                ipais_max = int(df_client.INST_PAY_AMT_INSTALMENT_SUM_MAX.values)
                                ipais_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                    'INST_PAY_AMT_INSTALMENT_SUM'].values)
                                ipais_axis_min = min(ipais_min, ipais_client)
                                ipais_axis_max = max(ipais_max, ipais_client)
                                
                                cond = ipais_client == ipais_min and ipais_min == ipais_q25 and \
                                    ipais_q25 == ipais_mean and ipais_mean == ipais_q75 and \
                                    ipais_q75 == ipais_max
                                
                                if not cond:
                                    fig_ipais = go.Figure()
                                
                                    fig_ipais.add_trace(go.Indicator(
                                        mode = "number+gauge+delta",
                                        value = ipais_client,
                                        delta = {'reference': ipais_mean,
                                                 'increasing': {'color': 'Crimson'},
                                                 'decreasing': {'color': 'Green'}},
                                        domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                        title = {'text': 'INST_PAY_AMT_INSTALMENT_SUM',
                                                 'font': {'size': 9}, 'align' : 'left'},
                                        gauge = {
                                            'shape': 'bullet',
                                            'axis': {'range': [ipais_axis_min, ipais_axis_max]},
                                            'threshold': {
                                                'line': {'color': 'black', 'width': 3},
                                                'thickness': 0.75,
                                                'value': ipais_client},
                                            'steps': [
                                                {'range': [0, ipais_min], 'color': 'white'},
                                                {'range': [ipais_min, ipais_q25], 'color': '#de3a5b'},
                                                {'range': [ipais_q25, ipais_mean], 'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [ipais_mean, ipais_q75],'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [ipais_q75, ipais_max], 'color': '#de3a5b'}],
                                            'bar': {'color': 'black'}}))

                                    fig_ipais.update_layout(height=200,
                                                            margin={'t':0, 'b':0, 'l':0})
  
                                    st.markdown(html_INST_PAY_AMT_INSTALMENT_SUM, unsafe_allow_html=True)
        
                                    # Go Indicator bullets
                                    st.plotly_chart(fig_ipais) 

                                    # ==================== ViolinPlot ========================================================
                                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='INST_PAY_AMT_INSTALMENT_SUM',
                                                   data=df_dashboard,
                                                   palette=['Green', 'Crimson'])
    
                                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                             ipais_client,
                                             color="orange",
                                             marker="$\\bigotimes$", markersize=28)
                                    plt.xlabel('TARGET', fontsize=16)
                                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                           linestyle='None',
                                                           markersize=16, label='Position du client')
                                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()
                                        
                                    # ==================== DistPlot ==========================================================
                                    # Non-défaillants
                                    sns.distplot(df_dashboard['INST_PAY_AMT_INSTALMENT_SUM'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 0],
                                                 label='Non-Défaillants', hist=False, color='Green')
                                    # Défaillants
                                    sns.distplot(df_dashboard['INST_PAY_AMT_INSTALMENT_SUM'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 1],
                                                 label='Défaillants', hist=False, color='Crimson')
                                    plt.xlabel('INST_PAY_AMT_INSTALMENT_SUM', fontsize=16)
                                    plt.ylabel('Probability Density', fontsize=16)
                                    plt.xticks(fontsize=16, rotation=90)
                                    plt.yticks(fontsize=16)
                                    # Position du client
                                    plt.axvline(x=ipais_client, color='orange', label='Position du client')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()                                   
                                    
                                else:
                                    
                                    st.markdown(html_INST_PAY_AMT_INSTALMENT_SUM, unsafe_allow_html=True)                                   
                                    st.write("Toutes les valeurs sont identiques") 


                        # ==============================================================
                        # Variable INST_PAY_DAYS_PAYMENT_RATIO_MAX
                        # Valeur maximal dans l'historique des précédents crédits remboursés
                        # dans Home Crédit du ratio : La date à laquelle le versement du crédit
                        # précédent était censé être payé (par rapport à la date de demande du
                        # prêt actuel) \ Quand les échéances du crédit précédent ont-elles été
                        # effectivement payées (par rapport à la date de demande du prêt 
                        # ==============================================================
                        if 'INST_PAY_DAYS_PAYMENT_RATIO_MAX' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif INST_PAY_DAYS_PAYMENT_RATIO_MAX...**'):
                                 
                                ipdprm_min = int(df_client.INST_PAY_DAYS_PAYMENT_RATIO_MAX_MIN.values)
                                ipdprm_q25 = int(df_client.INST_PAY_DAYS_PAYMENT_RATIO_MAX_Q25.values)
                                ipdprm_mean = int(df_client.INST_PAY_DAYS_PAYMENT_RATIO_MAX_MEAN.values)
                                ipdprm_q75 = int(df_client.INST_PAY_DAYS_PAYMENT_RATIO_MAX_Q75.values)
                                ipdprm_max = int(df_client.INST_PAY_DAYS_PAYMENT_RATIO_MAX_MAX.values)
                                ipdprm_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                    'INST_PAY_DAYS_PAYMENT_RATIO_MAX'].values)
                                ipdprm_axis_min = min(ipdprm_min, ipdprm_client)
                                ipdprm_axis_max = max(ipdprm_max, ipdprm_client)
                                
                                cond = ipdprm_client == ipdprm_min and ipdprm_min == ipdprm_q25 and \
                                    ipdprm_q25 == ipdprm_mean and ipdprm_mean == ipdprm_q75 and \
                                    ipdprm_q75 == ipdprm_max
                                
                                if not cond:
                                    fig_ipdprm = go.Figure()
                                
                                    fig_ipdprm.add_trace(go.Indicator(
                                        mode = "number+gauge+delta",
                                        value = ipdprm_client,
                                        delta = {'reference': ipdprm_mean,
                                                 'increasing': {'color': 'Crimson'},
                                                 'decreasing': {'color': 'Green'}},
                                        domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                        title = {'text': 'INST_PAY_DAYS_PAYMENT_RATIO_MAX',
                                                 'font': {'size': 9}, 'align' : 'left'},
                                        gauge = {
                                            'shape': 'bullet',
                                            'axis': {'range': [ipdprm_axis_min, ipdprm_axis_max]},
                                            'threshold': {
                                                'line': {'color': 'black', 'width': 3},
                                                'thickness': 0.75,
                                                'value': ipdprm_client},
                                            'steps': [
                                                {'range': [0, ipdprm_min], 'color': 'white'},
                                                {'range': [ipdprm_min, ipdprm_q25], 'color': '#de3a5b'},
                                                {'range': [ipdprm_q25, ipdprm_mean], 'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [ipdprm_mean, ipdprm_q75],'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [ipdprm_q75, ipdprm_max], 'color': '#de3a5b'}],
                                            'bar': {'color': 'black'}}))

                                    fig_ipdprm.update_layout(height=200,
                                                             margin={'t':0, 'b':0, 'l':0})
  
                                    st.markdown(html_INST_PAY_DAYS_PAYMENT_RATIO_MAX, unsafe_allow_html=True)
        
                                    # Go Indicator bullets
                                    st.plotly_chart(fig_ipdprm) 

                                    # ==================== ViolinPlot ========================================================
                                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='INST_PAY_DAYS_PAYMENT_RATIO_MAX',
                                                   data=df_dashboard,
                                                   palette=['Green', 'Crimson'])
    
                                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                             ipdprm_client,
                                             color="orange",
                                             marker="$\\bigotimes$", markersize=28)
                                    plt.xlabel('TARGET', fontsize=16)
                                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                           linestyle='None',
                                                           markersize=16, label='Position du client')
                                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()
                                        
                                    # ==================== DistPlot ==========================================================
                                    # Non-défaillants
                                    sns.distplot(df_dashboard['INST_PAY_DAYS_PAYMENT_RATIO_MAX'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 0],
                                                 label='Non-Défaillants', hist=False, color='Green')
                                    # Défaillants
                                    sns.distplot(df_dashboard['INST_PAY_DAYS_PAYMENT_RATIO_MAX'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 1],
                                                 label='Défaillants', hist=False, color='Crimson')
                                    plt.xlabel('INST_PAY_DAYS_PAYMENT_RATIO_MAX', fontsize=16)
                                    plt.ylabel('Probability Density', fontsize=16)
                                    plt.xticks(fontsize=16, rotation=90)
                                    plt.yticks(fontsize=16)
                                    # Position du client
                                    plt.axvline(x=ipdprm_client, color='orange', label='Position du client')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()                                   
                                                                        
                                else:
                                    
                                    st.markdown(html_INST_PAY_DAYS_PAYMENT_RATIO_MAX, unsafe_allow_html=True)                                   
                                    st.write("Toutes les valeurs sont identiques") 


                        # ==============================================================
                        # Variable POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM
                        # Somme des contrats actifs au cours du mois 
                        # ==============================================================
                        if 'POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM...**'):
                                                                 
                                pcncsas_min = int(df_client.POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM_MIN.values)
                                pcncsas_q25 = int(df_client.POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM_Q25.values)
                                pcncsas_mean = int(df_client.POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM_MEAN.values)
                                pcncsas_q75 = int(df_client.POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM_Q75.values)
                                pcncsas_max = int(df_client.POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM_MAX.values)
                                pcncsas_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                    'POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM'].values)
                                pcncsas_axis_min = min(pcncsas_min, pcncsas_client)
                                pcncsas_axis_max = max(pcncsas_max, pcncsas_client)
                                
                                cond = pcncsas_client == pcncsas_min and pcncsas_min == pcncsas_q25 and \
                                    pcncsas_q25 == pcncsas_mean and pcncsas_mean == pcncsas_q75 and \
                                    pcncsas_q75 == pcncsas_max
                                
                                if not cond:
                                    fig_pcncsas = go.Figure()
                                
                                    fig_pcncsas.add_trace(go.Indicator(
                                        mode = "number+gauge+delta",
                                        value = pcncsas_client,
                                        delta = {'reference': pcncsas_mean,
                                                 'increasing': {'color': 'Crimson'},
                                                 'decreasing': {'color': 'Green'}},
                                        domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                        title = {'text': 'POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM',
                                                 'font': {'size': 9}, 'align' : 'left'},
                                        gauge = {
                                            'shape': 'bullet',
                                            'axis': {'range': [pcncsas_axis_min, pcncsas_axis_max]},
                                            'threshold': {
                                                'line': {'color': 'black', 'width': 3},
                                                'thickness': 0.75,
                                                'value': pcncsas_client},
                                            'steps': [
                                                {'range': [0, pcncsas_min], 'color': 'white'},
                                                {'range': [pcncsas_min, pcncsas_q25], 'color': '#de3a5b'},
                                                {'range': [pcncsas_q25, pcncsas_mean], 'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [pcncsas_mean, pcncsas_q75],'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [pcncsas_q75, pcncsas_max], 'color': '#de3a5b'}],
                                            'bar': {'color': 'black'}}))

                                    fig_pcncsas.update_layout(height=200,
                                                              margin={'t':0, 'b':0, 'l':0})
  
                                    st.markdown(html_POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM, unsafe_allow_html=True)
        
                                    # Go Indicator bullets
                                    st.plotly_chart(fig_pcncsas) 

                                    # ==================== ViolinPlot ========================================================
                                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM',
                                                   data=df_dashboard,
                                                   palette=['Green', 'Crimson'])
    
                                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                             pcncsas_client,
                                             color="orange",
                                             marker="$\\bigotimes$", markersize=28)
                                    plt.xlabel('TARGET', fontsize=16)
                                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                           linestyle='None',
                                                           markersize=16, label='Position du client')
                                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()
                                        
                                    # ==================== DistPlot ==========================================================
                                    # Non-défaillants
                                    sns.distplot(df_dashboard['POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 0],
                                                 label='Non-Défaillants', hist=False, color='Green')
                                    # Défaillants
                                    sns.distplot(df_dashboard['POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 1],
                                                 label='Défaillants', hist=False, color='Crimson')
                                    plt.xlabel('POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM', fontsize=16)
                                    plt.ylabel('Probability Density', fontsize=16)
                                    plt.xticks(fontsize=16, rotation=90)
                                    plt.yticks(fontsize=16)
                                    # Position du client
                                    plt.axvline(x=pcncsas_client, color='orange', label='Position du client')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()                                   
                                                                                                            
                                else:
                                    
                                    st.markdown(html_POS_CASH_NAME_CONTRACT_STATUS_ACTIVE_SUM, unsafe_allow_html=True)                                   
                                    st.write("Toutes les valeurs sont identiques") 


                        # ==============================================================
                        # Variable PREV_APP_INTEREST_SHARE_MAX
                        # La valeur maximale de tous les précédents crédit dans d'autres
                        # institution : de la durée du crédit multiplié par l'annuité du prêt
                        # moins le montant final du crédit 
                        # ==============================================================
                        if 'PREV_APP_INTEREST_SHARE_MAX' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif PREV_APP_INTEREST_SHARE_MAX...**'):
                                                                                                 
                                paism_min = int(df_client.PREV_APP_INTEREST_SHARE_MAX_MIN.values)
                                paism_q25 = int(df_client.PREV_APP_INTEREST_SHARE_MAX_Q25.values)
                                paism_mean = int(df_client.PREV_APP_INTEREST_SHARE_MAX_MEAN.values)
                                paism_q75 = int(df_client.PREV_APP_INTEREST_SHARE_MAX_Q75.values)
                                paism_max = int(df_client.PREV_APP_INTEREST_SHARE_MAX_MAX.values)
                                paism_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                    'PREV_APP_INTEREST_SHARE_MAX'].values)
                                paism_axis_min = min(paism_min, paism_client)
                                paism_axis_max = max(paism_max, paism_client)
                                
                                cond = paism_client == paism_min and paism_min == paism_q25 and \
                                    paism_q25 == paism_mean and paism_mean == paism_q75 and \
                                    paism_q75 == paism_max
                                
                                if not cond:
                                    fig_paism = go.Figure()
                                
                                    fig_paism.add_trace(go.Indicator(
                                        mode = "number+gauge+delta",
                                        value = paism_client,
                                        delta = {'reference': paism_mean,
                                                 'increasing': {'color': 'Crimson'},
                                                 'decreasing': {'color': 'Green'}},
                                        domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                        title = {'text': 'PREV_APP_INTEREST_SHARE_MAX',
                                                 'font': {'size': 9}, 'align' : 'left'},
                                        gauge = {
                                            'shape': 'bullet',
                                            'axis': {'range': [paism_axis_min, paism_axis_max]},
                                            'threshold': {
                                                'line': {'color': 'black', 'width': 3},
                                                'thickness': 0.75,
                                                'value': paism_client},
                                            'steps': [
                                                {'range': [0, paism_min], 'color': 'white'},
                                                {'range': [paism_min, paism_q25], 'color': '#de3a5b'},
                                                {'range': [paism_q25, paism_mean], 'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [paism_mean, paism_q75],'color': '#dec7cb',
                                                 'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [paism_q75, paism_max], 'color': '#de3a5b'}],
                                            'bar': {'color': 'black'}}))

                                    fig_paism.update_layout(height=200,
                                                            margin={'t':0, 'b':0, 'l':0})
  
                                    st.markdown(html_PREV_APP_INTEREST_SHARE_MAX, unsafe_allow_html=True)
        
                                    # Go Indicator bullets
                                    st.plotly_chart(fig_paism) 

                                    # ==================== ViolinPlot ========================================================
                                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='PREV_APP_INTEREST_SHARE_MAX',
                                                   data=df_dashboard,
                                                   palette=['Green', 'Crimson'])
    
                                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                             paism_client,
                                             color="orange",
                                             marker="$\\bigotimes$", markersize=28)
                                    plt.xlabel('TARGET', fontsize=16)
                                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                           linestyle='None',
                                                           markersize=16, label='Position du client')
                                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()
                                        
                                    # ==================== DistPlot ==========================================================
                                    # Non-défaillants
                                    sns.distplot(df_dashboard['PREV_APP_INTEREST_SHARE_MAX'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 0],
                                                 label='Non-Défaillants', hist=False, color='Green')
                                    # Défaillants
                                    sns.distplot(df_dashboard['PREV_APP_INTEREST_SHARE_MAX'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 1],
                                                 label='Défaillants', hist=False, color='Crimson')
                                    plt.xlabel('PREV_APP_INTEREST_SHARE_MAX', fontsize=16)
                                    plt.ylabel('Probability Density', fontsize=16)
                                    plt.xticks(fontsize=16, rotation=90)
                                    plt.yticks(fontsize=16)
                                    # Position du client
                                    plt.axvline(x=paism_client, color='orange', label='Position du client')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.pyplot()                                   
                                                                                                                                                
                                else:
                                    
                                    st.markdown(html_PREV_APP_INTEREST_SHARE_MAX, unsafe_allow_html=True)                                   
                                    st.write("Toutes les valeurs sont identiques") 

                        # ==============================================================
                        # Variable CODE_GENDER
                        # Sexe 
                        # ==============================================================
                        if 'CODE_GENDER' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif CODE_GENDER...**'):
                                    
                                    cg_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                        'CODE_GENDER'].values)

                                    st.markdown(html_CODE_GENDER, unsafe_allow_html=True)
                                    
                                    # ==================== ViolinPlot ========================================================
                                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='CODE_GENDER',
                                                   data=df_dashboard,
                                                   palette=['Green', 'Crimson'])
    
                                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                             cg_client,
                                             color="orange",
                                             marker="$\\bigotimes$", markersize=28)
                                    plt.xlabel('TARGET', fontsize=16)
                                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                           linestyle='None',
                                                           markersize=16, label='Position du client')
                                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    st.pyplot()
                                        
                                    # ==================== DistPlot ==========================================================
                                    # Non-défaillants
                                    sns.distplot(df_dashboard['CODE_GENDER'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 0],
                                                 label='Non-Défaillants', hist=False, color='Green')
                                    # Défaillants
                                    sns.distplot(df_dashboard['CODE_GENDER'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 1],
                                                 label='Défaillants', hist=False, color='Crimson')
                                    plt.xlabel('CODE_GENDER', fontsize=16)
                                    plt.ylabel('Probability Density', fontsize=16)
                                    plt.xticks(fontsize=16, rotation=90)
                                    plt.yticks(fontsize=16)
                                    # Position du client
                                    plt.axvline(x=cg_client, color='orange', label='Position du client')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                    st.pyplot()   
                                    
                                    
                        # ==============================================================
                        # Variable FLAG_OWN_CAR
                        # Indicateur si le client possède une voiture
                        # ==============================================================
                        if 'FLAG_OWN_CAR' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif FLAG_OWN_CAR...**'):
                                
                                    foc_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                        'FLAG_OWN_CAR'].values)

                                    st.markdown(html_FLAG_OWN_CAR, unsafe_allow_html=True)
                                    
                                    # ==================== ViolinPlot ========================================================
                                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='FLAG_OWN_CAR',
                                                   data=df_dashboard,
                                                   palette=['Green', 'Crimson'])
    
                                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                             foc_client,
                                             color="orange",
                                             marker="$\\bigotimes$", markersize=28)
                                    plt.xlabel('TARGET', fontsize=16)
                                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                           linestyle='None',
                                                           markersize=16, label='Position du client')
                                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    st.pyplot()
                                        
                                    # ==================== DistPlot ==========================================================
                                    # Non-défaillants
                                    sns.distplot(df_dashboard['FLAG_OWN_CAR'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 0],
                                                 label='Non-Défaillants', hist=False, color='Green')
                                    # Défaillants
                                    sns.distplot(df_dashboard['FLAG_OWN_CAR'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 1],
                                                 label='Défaillants', hist=False, color='Crimson')
                                    plt.xlabel('FLAG_OWN_CAR', fontsize=16)
                                    plt.ylabel('Probability Density', fontsize=16)
                                    plt.xticks(fontsize=16, rotation=90)
                                    plt.yticks(fontsize=16)
                                    # Position du client
                                    plt.axvline(x=foc_client, color='orange', label='Position du client')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                    st.pyplot()   


                        # ==============================================================
                        # Variable NAME_EDUCATION_TYPE_HIGHER_EDUCATION
                        # Indicateur si le client possède une voiture
                        # ==============================================================
                        if 'NAME_EDUCATION_TYPE_HIGHER_EDUCATION' in feat_imp_to_show:
                                
                            with st.spinner('**Chargement du graphique comparatif NAME_EDUCATION_TYPE_HIGHER_EDUCATION...**'):
                                
                                    nethe_client = int(df_dashboard[df_dashboard['SK_ID_CURR'] == client_id][
                                        'NAME_EDUCATION_TYPE_HIGHER_EDUCATION'].values)

                                    st.markdown(html_NAME_EDUCATION_TYPE_HIGHER_EDUCATION, unsafe_allow_html=True)
                                    
                                    # ==================== ViolinPlot ========================================================
                                    sns.violinplot(x='PRED_CLASSE_CLIENT', y='NAME_EDUCATION_TYPE_HIGHER_EDUCATION',
                                                   data=df_dashboard,
                                                   palette=['Green', 'Crimson'])
    
                                    plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                             nethe_client,
                                             color="orange",
                                             marker="$\\bigotimes$", markersize=28)
                                    plt.xlabel('TARGET', fontsize=16)
                                    client = mlines.Line2D([], [], color='orange', marker='$\\bigotimes$',
                                                           linestyle='None',
                                                           markersize=16, label='Position du client')
                                    plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    st.pyplot()
                                        
                                    # ==================== DistPlot ==========================================================
                                    # Non-défaillants
                                    sns.distplot(df_dashboard['NAME_EDUCATION_TYPE_HIGHER_EDUCATION'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 0],
                                                 label='Non-Défaillants', hist=False, color='Green')
                                    # Défaillants
                                    sns.distplot(df_dashboard['NAME_EDUCATION_TYPE_HIGHER_EDUCATION'][df_dashboard[
                                        'PRED_CLASSE_CLIENT'] == 1],
                                                 label='Défaillants', hist=False, color='Crimson')
                                    plt.xlabel('NAME_EDUCATION_TYPE_HIGHER_EDUCATION', fontsize=16)
                                    plt.ylabel('Probability Density', fontsize=16)
                                    plt.xticks(fontsize=16, rotation=90)
                                    plt.yticks(fontsize=16)
                                    # Position du client
                                    plt.axvline(x=nethe_client, color='orange', label='Position du client')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
                                    st.pyplot()  


# --------------------------------------------------------------------
# FACTEURS D'INFLUENCE : SHAP VALUE
# --------------------------------------------------------------------
    
def affiche_facteurs_influence():
    ''' Affiche les facteurs d'influence du client courant
    '''
    html_facteurs_influence="""
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #DEC7CB; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:#DEC7CB; color:Crimson;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                       DETERMINANTS DU PROFIL CLIENT
                  </h3>
            </div>
        </div>
        """
    
    # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES =========================== 
    if st.sidebar.checkbox("Voir facteurs d\'influence"):     
        
        st.markdown(html_facteurs_influence, unsafe_allow_html=True)

        with st.spinner('**Affiche les facteurs d\'influence du client courant...**'):                 
                       
            with st.expander('Facteurs d\'influence du client courant',
                              expanded=True):
                
                explainer = shap.TreeExplainer(best_model)
                
                client_index = test_set[test_set['SK_ID_CURR'] == client_id].index.item()
                X_shap = test_set.set_index('SK_ID_CURR')
                X_test_courant = X_shap.iloc[client_index]
                X_test_courant_array = X_test_courant.values.reshape(1, -1)
                
                shap_values_courant = explainer.shap_values(X_test_courant_array)
                
                col1, col2 = st.columns([1, 1])
                # BarPlot du client courant
                with col1:

                    plt.clf()
                    

                    # BarPlot du client courant
                    shap.plots.bar(bar_values[client_index], max_display=40)
                    
                    fig = plt.gcf()
                    fig.set_size_inches((10, 20))
                    # Plot the graph on the dashboard
                    st.pyplot(fig)
     
                # Décision plot du client courant
                with col2:
                    plt.clf()

                    # Décision Plot
                    shap.decision_plot(explainer.expected_value[1], shap_values_courant[1],
                                    X_test_courant)
                
                    fig2 = plt.gcf()
                    fig2.set_size_inches((10, 15))
                    # Plot the graph on the dashboard
                    st.pyplot(fig2)
                    
st.sidebar.subheader('Facteurs d\'influence')
affiche_facteurs_influence()

# --------------------------------------------------------------------
# STATISTIQUES GENERALES
# --------------------------------------------------------------------

dico_stats = {'Target': 'TARGET',
              'Type de prêt': 'NAME_CONTRACT_TYPE',
              'Sexe': 'CODE_GENDER',
              'Tél. professionnel': 'FLAG_EMP_PHONE',
              'Note région où vit client': 'REGION_RATING_CLIENT_W_CITY',
              'Niveau éducation du client': 'NAME_EDUCATION_TYPE',
              'Profession du client': 'OCCUPATION_TYPE',
              'Type d\'organisation de travail du client': 'ORGANIZATION_TYPE',
              'Adresse du client = adresse de contact': 'REG_CITY_NOT_LIVE_CITY',
              'Région du client = adresse professionnelle': 'REG_CITY_NOT_WORK_CITY',
              'Adresse du client = adresse professionnelle': 'LIVE_CITY_NOT_WORK_CITY',
              'Logement du client': 'NAME_HOUSING_TYPE',
              'Statut familial': 'NAME_FAMILY_STATUS',
              'Type de revenu du client': 'NAME_INCOME_TYPE',
              'Client possède une maison ou appartement?': 'FLAG_OWN_REALTY',
              'Accompagnateur lors de la demande de prêt?': 'NAME_TYPE_SUITE',
              'Quel jour de la semaine le client a-t-il demandé le prêt ?': 'WEEKDAY_APPR_PROCESS_START',
              'Le client a-t-il fourni un numéro de téléphone portable ?': 'FLAG_MOBIL',
              'Le client a-t-il fourni un numéro de téléphone professionnel fixe ?': 'FLAG_WORK_PHONE',
              'Le téléphone portable était-il joignable?': 'FLAG_CONT_MOBILE',             
              'Le client a-t-il fourni un numéro de téléphone domicile fixe ?': 'FLAG_PHONE',
              'Le client a-t-il fourni une adresse électronique': 'FLAG_EMAIL',
              'Âge (ans)': 'AGE_YEARS',
              'Combien d\'années avant la demande la personne a commencé son emploi actuel ?': 'YEARS_EMPLOYED',
              'Combien de jours avant la demande le client a-t-il changé son enregistrement ?': 'DAYS_REGISTRATION',
              'Combien de jours avant la demande le client a-t-il changé la pièce d\'identité avec laquelle il a demandé le prêt ?': 'DAYS_ID_PUBLISH',
              'Prix du bien que le client a demandé': 'AMT_GOODS_PRICE',
              'Nombre d\enfants?': 'CNT_CHILDREN',
              'Revenu du client': 'AMT_INCOME_TOTAL',
              'Montant du crédit du prêt': 'AMT_CREDIT',
              'Annuité de prêt': 'AMT_ANNUITY',
              'Âge de la voiture du client': 'OWN_CAR_AGE',
              'Combien de membres de la famille a le client': 'CNT_FAM_MEMBERS',
              'Population normalisée de la région où vit le client': 'REGION_POPULATION_RELATIVE',
              'Notre évaluation de la région où vit le client (1 ou 2 ou 3)': 'REGION_RATING_CLIENT',
              'Indicateur si l\'adresse permanente du client ne correspond pas à l\'adresse de contact': 'REG_REGION_NOT_LIVE_REGION',
              'Indicateur si l\'adresse permanente du client ne correspond pas à l\'adresse professionnelle': 'REG_REGION_NOT_WORK_REGION',
              'Indicateur si l\'adresse de contact du client ne correspond pas à l\'adresse de travail': 'LIVE_REGION_NOT_WORK_REGION',
              'Combien de jours avant la demande le client a-t-il changé de téléphone ?': 'DAYS_LAST_PHONE_CHANGE',
              'Statut des crédits déclarés par le Credit Bureau': 'CREDIT_ACTIVE',
              'Devise recodée du crédit du Credit Bureau': 'CREDIT_CURRENCY',
              'Type de crédit du Bureau de crédit (voiture ou argent liquide...)': 'CREDIT_TYPE',
              'Combien d\années avant la demande actuelle le client a-t-il demandé un crédit au Credit Bureau ?': 'YEARS_CREDIT',
              'Durée restante du crédit CB (en jours) au moment de la demande dans Crédit immobilier': 'DAYS_CREDIT_ENDDATE',
              'Combien de jours avant la demande de prêt la dernière information sur la solvabilité du Credit Bureau a-t-elle été fournie ?': 'DAYS_CREDIT_UPDATE',
              'Nombre de jours de retard sur le crédit CB au moment de la demande de prêt': 'CREDIT_DAY_OVERDUE',
              'Montant maximal des impayés sur le crédit du Credit Bureau jusqu\'à présent': 'AMT_CREDIT_MAX_OVERDUE',
              'Combien de fois le crédit du Bureau de crédit a-t-il été prolongé ?': 'CNT_CREDIT_PROLONG',
              'Montant actuel du crédit du Credit Bureau': 'AMT_CREDIT_SUM',
              'Dette actuelle sur le crédit du Credit Bureau': 'AMT_CREDIT_SUM_DEBT',
              'Limite de crédit actuelle de la carte de crédit déclarée dans le Bureau de crédit': 'AMT_CREDIT_SUM_LIMIT',
              'Montant actuel en retard sur le crédit du Bureau de crédit': 'AMT_CREDIT_SUM_OVERDUE',
              'Annuité du crédit du Credit Bureau': 'AMT_ANNUITY',
              'Statut du prêt du Credit Bureau durant le mois': 'STATUS',
              'Mois du solde par rapport à la date de la demande': 'MONTHS_BALANCE',
              'Statut du contrat au cours du mois': 'NAME_CONTRACT_STATUS',
              'Solde au cours du mois du crédit précédent': 'AMT_BALANCE',
              'Montant total à recevoir sur le crédit précédent': 'AMT_TOTAL_RECEIVABLE',
              'Nombre d\'échéances payées sur le crédit précédent': 'CNT_INSTALMENT_MATURE_CUM',
              'Mois du solde par rapport à la date d\'application': 'MONTHS_BALANCE',
              'Limite de la carte de crédit au cours du mois du crédit précédent': 'AMT_CREDIT_LIMIT_ACTUAL',
              'Montant retiré au guichet automatique pendant le mois du crédit précédent': 'AMT_DRAWINGS_ATM_CURRENT',
              'Montant prélevé au cours du mois du crédit précédent': 'AMT_DRAWINGS_CURRENT',
              'Montant des autres prélèvements au cours du mois du crédit précédent': 'AMT_DRAWINGS_OTHER_CURRENT',
              'Montant des prélèvements ou des achats de marchandises au cours du mois de la crédibilité précédente': 'AMT_DRAWINGS_POS_CURRENT',
              'Versement minimal pour ce mois du crédit précédent': 'AMT_INST_MIN_REGULARITY',
              'Combien le client a-t-il payé pendant le mois sur le crédit précédent ?': 'AMT_PAYMENT_CURRENT',
              'Combien le client a-t-il payé au total pendant le mois sur le crédit précédent ?': 'AMT_PAYMENT_TOTAL_CURRENT',
              'Montant à recevoir pour le principal du crédit précédent': 'AMT_RECEIVABLE_PRINCIPAL',
              'Montant à recevoir sur le crédit précédent': 'AMT_RECIVABLE', 
              'Nombre de retraits au guichet automatique durant ce mois sur le crédit précédent': 'CNT_DRAWINGS_ATM_CURRENT',
              'Nombre de retraits pendant ce mois sur le crédit précédent': 'CNT_DRAWINGS_CURRENT',
              'Nombre d\'autres retraits au cours de ce mois sur le crédit précédent': 'CNT_DRAWINGS_OTHER_CURRENT',
              'Nombre de retraits de marchandises durant ce mois sur le crédit précédent': 'CNT_DRAWINGS_POS_CURRENT',
              'DPD (jours de retard) au cours du mois sur le crédit précédent': 'SK_DPD',
              'DPD (Days past due) au cours du mois avec tolérance (les dettes avec de faibles montants de prêt sont ignorées) du crédit précédent': 'SK_DPD_DEF',
              'La date à laquelle le versement du crédit précédent était censé être payé (par rapport à la date de demande du prêt actuel)': 'DAYS_INSTALMENT',
              'Quand les échéances du crédit précédent ont-elles été effectivement payées (par rapport à la date de demande du prêt actuel) ?': 'DAYS_ENTRY_PAYMENT',
              'Version du calendrier des versements (0 pour la carte de crédit) du crédit précédent': 'NUM_INSTALMENT_VERSION',
              'Sur quel versement nous observons le paiement': 'NUM_INSTALMENT_NUMBER',
              'Quel était le montant de l\'acompte prescrit du crédit précédent sur cet acompte ?': 'AMT_INSTALMENT',
              'Ce que le client a effectivement payé sur le crédit précédent pour ce versement': 'AMT_PAYMENT',
              'Statut du contrat au cours du mois': 'NAME_CONTRACT_STATUS',
              'Durée du crédit précédent (peut changer avec le temps)': 'CNT_INSTALMENT',
              'Versements restant à payer sur le crédit précédent': 'CNT_INSTALMENT_FUTURE',
              'EXT_SOURCE_1': 'EXT_SOURCE_1',
              'EXT_SOURCE_2': 'EXT_SOURCE_2',
              'EXT_SOURCE_3': 'EXT_SOURCE_3',
              'FLAG_DOCUMENT_2': 'FLAG_DOCUMENT_2',
              'FLAG_DOCUMENT_3': 'FLAG_DOCUMENT_3',
              'FLAG_DOCUMENT_4': 'FLAG_DOCUMENT_4',
              'FLAG_DOCUMENT_5': 'FLAG_DOCUMENT_5',
              'FLAG_DOCUMENT_6': 'FLAG_DOCUMENT_6',
              'FLAG_DOCUMENT_7': 'FLAG_DOCUMENT_7',
              'FLAG_DOCUMENT_8': 'FLAG_DOCUMENT_8',
              'FLAG_DOCUMENT_9': 'FLAG_DOCUMENT_9',
              'FLAG_DOCUMENT_10': 'FLAG_DOCUMENT_10',
              'FLAG_DOCUMENT_11': 'FLAG_DOCUMENT_11',
              'FLAG_DOCUMENT_12': 'FLAG_DOCUMENT_12',
              'FLAG_DOCUMENT_13': 'FLAG_DOCUMENT_13',
              'FLAG_DOCUMENT_14': 'FLAG_DOCUMENT_14',
              'FLAG_DOCUMENT_15': 'FLAG_DOCUMENT_15',
              'FLAG_DOCUMENT_16': 'FLAG_DOCUMENT_16',
              'FLAG_DOCUMENT_17': 'FLAG_DOCUMENT_17',
              'FLAG_DOCUMENT_18': 'FLAG_DOCUMENT_18',
              'FLAG_DOCUMENT_19': 'FLAG_DOCUMENT_19',
              'FLAG_DOCUMENT_20': 'FLAG_DOCUMENT_20',
              'FLAG_DOCUMENT_21': 'FLAG_DOCUMENT_21',
              'FONDKAPREMONT_MODE': 'FONDKAPREMONT_MODE',
              'HOUSETYPE_MODE': 'HOUSETYPE_MODE',
              'WALLSMATERIAL_MODE': 'WALLSMATERIAL_MODE',
              'EMERGENCYSTATE_MODE': 'EMERGENCYSTATE_MODE',
              'FLOORSMAX_AVG': 'FLOORSMAX_AVG',
              'FLOORSMAX_MEDI': 'FLOORSMAX_MEDI',
              'FLOORSMAX_MODE': 'FLOORSMAX_MODE',
              'FLOORSMIN_MODE': 'FLOORSMIN_MODE',
              'FLOORSMIN_AVG': 'FLOORSMIN_AVG',
              'FLOORSMIN_MEDI': 'FLOORSMIN_MEDI',
              'APARTMENTS_AVG': 'APARTMENTS_AVG',
              'APARTMENTS_MEDI': 'APARTMENTS_MEDI',
              'APARTMENTS_MODE': 'APARTMENTS_MODE',
              'BASEMENTAREA_AVG': 'BASEMENTAREA_AVG',
              'BASEMENTAREA_MEDI': 'BASEMENTAREA_MEDI',
              'BASEMENTAREA_MODE': 'BASEMENTAREA_MODE',
              'YEARS_BEGINEXPLUATATION_AVG': 'YEARS_BEGINEXPLUATATION_AVG',
              'YEARS_BEGINEXPLUATATION_MODE': 'YEARS_BEGINEXPLUATATION_MODE',
              'YEARS_BEGINEXPLUATATION_MEDI': 'YEARS_BEGINEXPLUATATION_MEDI',
              'YEARS_BUILD_AVG': 'YEARS_BUILD_AVG',
              'YEARS_BUILD_MODE': 'YEARS_BUILD_MODE',
              'YEARS_BUILD_MEDI': 'YEARS_BUILD_MEDI',
              'COMMONAREA_AVG': 'COMMONAREA_AVG',
              'COMMONAREA_MEDI': 'COMMONAREA_MEDI',
              'COMMONAREA_MODE': 'COMMONAREA_MODE',
              'ELEVATORS_AVG': 'ELEVATORS_AVG',
              'ELEVATORS_MODE': 'ELEVATORS_MODE',
              'ELEVATORS_MEDI': 'ELEVATORS_MEDI',
              'ENTRANCES_AVG': 'ENTRANCES_AVG',
              'ENTRANCES_MODE': 'ENTRANCES_MODE',
              'ENTRANCES_MEDI': 'ENTRANCES_MEDI',
              'LANDAREA_AVG': 'LANDAREA_AVG',
              'LANDAREA_MEDI': 'LANDAREA_MEDI',
              'LANDAREA_MODE': 'LANDAREA_MODE',
              'LIVINGAPARTMENTS_AVG': 'LIVINGAPARTMENTS_AVG',
              'LIVINGAPARTMENTS_MODE': 'LIVINGAPARTMENTS_MODE',
              'LIVINGAPARTMENTS_MEDI': 'LIVINGAPARTMENTS_MEDI',
              'LIVINGAREA_AVG': 'LIVINGAREA_AVG',
              'LIVINGAREA_MODE': 'LIVINGAREA_MODE',
              'NONLIVINGAPARTMENTS_AVG': 'NONLIVINGAPARTMENTS_AVG',
              'NONLIVINGAPARTMENTS_MODE': 'NONLIVINGAPARTMENTS_MODE',
              'NONLIVINGAREA_AVG': 'NONLIVINGAREA_AVG',
              'NONLIVINGAREA_MEDI': 'NONLIVINGAREA_MEDI',
              'TOTALAREA_MODE': 'TOTALAREA_MODE',
              'OBS_30_CNT_SOCIAL_CIRCLE': 'OBS_30_CNT_SOCIAL_CIRCLE',
              'DEF_30_CNT_SOCIAL_CIRCLE': 'DEF_30_CNT_SOCIAL_CIRCLE',
              'DEF_60_CNT_SOCIAL_CIRCLE': 'DEF_60_CNT_SOCIAL_CIRCLE',
              'AMT_REQ_CREDIT_BUREAU_HOUR': 'AMT_REQ_CREDIT_BUREAU_HOUR',
              'AMT_REQ_CREDIT_BUREAU_DAY': 'AMT_REQ_CREDIT_BUREAU_DAY',
              'AMT_REQ_CREDIT_BUREAU_WEEK': 'AMT_REQ_CREDIT_BUREAU_WEEK',
              'AMT_REQ_CREDIT_BUREAU_MON': 'AMT_REQ_CREDIT_BUREAU_MON',
              'AMT_REQ_CREDIT_BUREAU_QRT': 'AMT_REQ_CREDIT_BUREAU_QRT',
              'AMT_REQ_CREDIT_BUREAU_YEAR': 'AMT_REQ_CREDIT_BUREAU_YEAR',
              'DAYS_ENDDATE_FACT': 'DAYS_ENDDATE_FACT'}

path_img = "resources/images/stats/"
   
def affiche_stats():
    ''' Affiche les statistiques générales provenant de l'EDA
    '''
    html_facteurs_influence="""
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #DEC7CB; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:#DEC7CB; color:Crimson;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                     CLIENTS DEFAILLANTS
                  </h3>
            </div>
        </div>
        """
    
    # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES =========================== 
    if st.sidebar.checkbox("Distribution des données"):     
        
        st.markdown(html_facteurs_influence, unsafe_allow_html=True)

        with st.spinner('**Affiche les statistiques générales/pour les défaillants...**'):                 
                       
            with st.expander('Statistiques Descriptives',
                              expanded=False):
                choix = st.selectbox("Choisir une variable : ", dico_stats.keys()) 
                nom_img = dico_stats[choix]
                img =  Image.open(path_img + nom_img + ".png")                     
                st.image(img)       
                
st.sidebar.subheader('Description détaillée des données')
affiche_stats()


# Contact Form


with st.expander("Contact us"):
    with st.form(key='contact', clear_on_submit=True):
   
        
        email = st.text_input('Contact Email or Personal code')
        st.text_area("Requête","Merci de renseigner votre requête avec le maximum de détails")  
        
        submit_button = st.form_submit_button(label='Envoyer à l\'équipe technique')
        



# ====================================================================
# FOOTER
# ====================================================================

html_line="""
<br>
<br>
<br>
<br>
<hr style= "  display: block;
  margin-top: 0.5em;
  margin-bottom: 0.5em;
  margin-left: auto;
  margin-right: auto;
  border-style: inset;
  border-width: 1.5px;">
<p style="color:darkblue; text-align: right; font-size:14px;">Auteur : Badr ABOUFARASSE</p>
 <p style="color:darkblue; text-align: right; font-size:14px;">Contact: aboufirasse.badr@gmail.com</p>
"""
st.markdown(html_line, unsafe_allow_html=True)


