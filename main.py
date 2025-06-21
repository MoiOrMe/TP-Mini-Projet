from TP1 import *
from TP2 import *
from TP3 import *


def ChoixTP():
    choiceTP = -1
    print("Bienvenue dans les TP de Machine learning de Robin !")
    while choiceTP != 0:
        choiceTP = int(input("Quelle TP voulez-vous lancer ? :\n1. TP1 - PCA\n2. TP2 - Reconnaissance de Visages (Work in progress)\n3. TP3 - Méthodes de réduction de dimensionnalité t-SNE, UMAP\n0. Quitter\n"))
        if choiceTP == 1:
            ChoixExoTP1()
        elif choiceTP == 2:
            ChoixExoTP2()
        elif choiceTP == 3:
            ChoixExoTP3()
        elif choiceTP < 0 or choiceTP > 3:
            print("Choix invalide, veuillez réessayer.")


if __name__ == "__main__":
    ChoixTP()