import os
import streamlit as st 
import time
from fonctions import traitement, rapport
import pandas as pd
import numpy as np


st.header("_Machine Learning_ is :blue[cool]:sunglasses:", divider=True)
# Titre principal
st.title("Credit App")
st.write("Application web de prédiction du non remboursement de crédit")



# espacement des variables
st.markdown("<br><br>", unsafe_allow_html=True)  # 1) HTML : contrôlé (place 2 <br>)


# charger les donnees
data  = st.file_uploader("charger votre jeu de donnees", accept_multiple_files=False, type=["csv", "xlsx"])
if data:

    with st.spinner("chargement ..."):
        time.sleep(2)
    
    df, data_base, dico, dico_class = traitement(data)
    dd = pd.DataFrame(df)

    # Donnees pas encore traitées
    st.write("Datas de base", data_base)

    st.write("Datas encodées")
    st.dataframe(data=df)
    st.write("Dictionnaire", dico)

    
    st.write("Votre jeu de donnée est chargé")
    st.success("Done!")


    # Première question
 
    confusion_matric,classification_report,score, resultat, modele  = rapport(df) #, nouvelle_prediction=prediction)
  
    db = pd.DataFrame(classification_report).transpose()

    st.title("Matrice de confusion")
    st.write(confusion_matric)

    st.title("Classification report")
    st.dataframe(classification_report)

    st.write(f"$R²$ = {score} (coefficient de détermination)")

    st.title("Rapport de côte (odds-ratio)")
    st.dataframe(resultat)

    choix = st.selectbox("Voulez vous faire une prediction", ["oui", "non"], index=1)
    if choix == "oui":
        donnees_saisies = st.text_input("Saisir vos données (ex: 0,1,1,1,1,1,0,1)")

        if donnees_saisies:
            try:
                # Étape 1 : Séparer la chaîne et convertir en int
                valeurs = [int(x.strip()) for x in donnees_saisies.split(',')]

                # Étape 2 : Transformer en tableau numpy 2D
                array = np.array([valeurs])

                # Étape 3 : Prédiction avec ton modèle
                pred = modele.predict(array)

                # Affichage du résultat
                st.write(f"Prédiction : {pred[0]}")

                if pred[0] == 1:
                    st.warning("⚠️ Ce client risque de faire un défaut sur le crédit.")
                else:
                    st.success("✅ Ce client est peu susceptible de faire défaut.")
            
            except ValueError:
                st.error("Erreur : veuillez entrer uniquement des nombres séparés par des virgules (ex: 0,1,1,1,1,1,0,1).")
    else:
        st.write("Vous ferez sans doutes vos prédiction plus tard")
 
   
   
   

   