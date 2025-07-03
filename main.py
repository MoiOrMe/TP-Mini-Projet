from TP1 import *
from TP2 import *
from TP3 import *
from TP4 import *
from TP5 import *


def ChoixTP():
    choiceTP = -1
    print("Bienvenue dans les TP de Machine learning de Robin !")
    while choiceTP != 0:
        choiceTP = int(input("Quelle TP voulez-vous lancer ? :\n1. TP1 - PCA\n2. TP2 - Reconnaissance de Visages\n3. TP3 - Méthodes de réduction de dimensionnalité t-SNE, UMAP\n4. TP4 - Clustering non supervisé pour la segmentation d'images Méthodes : K-Means & Clustering hiérarchique\n5. TP5 - Apprentissage supervisé : Classifieurs, Validation, Biais et Variance\tMéthodes : KNN, Arbres de décision & Random Forest\n0. Quitter\n"))
        if choiceTP == 1:
            ChoixExoTP1()
        elif choiceTP == 2:
            ChoixExoTP2()
        elif choiceTP == 3:
            ChoixExoTP3()
        elif choiceTP == 4:
            ChoixExoTP4()
        elif choiceTP == 5:
            ChoixExoTP5()
        elif choiceTP < 0 or choiceTP > 5:
            print("Choix invalide, veuillez réessayer.")


if __name__ == "__main__":
    ChoixTP()