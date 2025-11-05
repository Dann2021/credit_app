# création d'un modèle de regression logistique
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# creation d'un fonction pour traiter les donnees
# cette fonction doit retourner un dataset pret et encoder
def traitement(chemin_data):

    if isinstance(chemin_data, list):
        # Imprimer les fichiers lus pour le débogage
        print("Liste de chemins de fichiers détectée :", chemin_data)
        # Si vous voulez lire plusieurs fichiers CSV et les combiner :
        data = pd.concat([pd.read_csv(file) for file in chemin_data], ignore_index=True)
    else:
        data = pd.read_csv(chemin_data)  # Cas d'un seul chemin de fichier

    # on lit le fichier de data
    #data = pd.read_csv(chemin_data)

    # on affiche l'entete
    data.head()

    # Enregistrement des colonnes
    cols = data.columns

    # Creation de nouvelles colonnes selon les variables que j'avais déjà definie
    nouvelles_colonnes = ["Y","X2","X3","X6","X9","X1","X7","X8","X4"]
    
    # creation d'un dictionnaire pour afficher la correspondance des nouvelles colonnes avec les anciennes
    dico = {}
    for colonne1, colonne2 in zip(cols[2:], nouvelles_colonnes):
        dico.update({colonne1:colonne2})
        
    # Suppression des colonnes inutiles
    col1 = cols[0] # Ici on supprime la colonne horodateur
    col2 = cols[1] # ici on supprime la colonne qui contient la question de savoir si un personne a oui ou non contracté un crédit
    data = data.drop([col1, col2], axis=1)

    # suppression des valeurs manquantes
    data = data.dropna()

    # on copie maintenant le dataframe pour renommer les variables
    df = data.copy()

    # on renomme maintenant les variables
    df.columns = nouvelles_colonnes

    # On encode maintenant les donnees

    # creation d'un dico pour voir les differentes de chaque colonnes
    dico_class = {}
    for col in nouvelles_colonnes:
        df[col], uniques = pd.factorize(df[col])
        dico_class.update({col:uniques})

    # Valeur renvoyée par la fonction
    return df, data, pd.Series(dico, name="variables"), pd.Series(dico_class)




# creation d'une fonction qui étudiera les donnees et renvera des resultat
def rapport(data_frame, nouvelle_prediction=None):

    # separation des donnees en couple (x,y)
    X, y = data_frame.drop("Y", axis=1), data_frame['Y']

    # creation du modele de regression logistiques
    modele = LogisticRegression(max_iter=1000)

    # divison des donnees en data de test et de train
    xtrain,xtest, ytrain, ytest = train_test_split(X,y, test_size=0.2, random_state=42)

    # entraînement du modele 
    modele.fit(xtrain,ytrain)

    # calcule pour faire les odds ratios
    # Coefficients
    coefficients = modele.coef_[0]

    # Noms des variables
    features = xtrain.columns

    # Calcul des odds ratios
    odds_ratios = np.exp(coefficients)

    # Tableau récapitulatif
    results = pd.DataFrame({
        'Variable': features,
        'Coefficient': coefficients,
        'Odds Ratio': odds_ratios
    })

    # prediction de test
    ypred = modele.predict(xtest)

    #print("Matrice de confusion")
    cm = confusion_matrix(ytest,ypred)
    #print(cm)

    #print("-----------------------------")
    #print("\nRapport de classification")
    cr = classification_report(ytest,ypred, output_dict=True)
    
    #print(cr)

    #print("-----------------------------")
    asc = accuracy_score(ytest, ypred)
    #print(f"Précision du modèle : {asc} ")
    #print(f"R^2 = {modele.score(xtest,ytest)}")

    # Prediction
    pred = None
    if nouvelle_prediction:
        pred = modele.predict(nouvelle_prediction)
        print(f"Prédiction : {pred}")

        # verification s'il y a defaut
        if pred[0] == 1:
            print("\nCe client risque de faire un défaut sur le crédit")
        else:
            print("\nCe client est peu susceptible de faire défaut")
    
    return cm, cr, asc,  results, modele
            

    
    
    