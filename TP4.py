def Exercice_1():
    print("Exercice 1 : Segmentation d'images par K-Means")


def Exercice_2():
    print("Exercice 2 : Clustering hiérarchique sur chiffres manuscrits")


def ChoixExoTP4():
    choiceExo = -1
    print("Bienvenue dans le TP de Méthodes de réduction de dimensionnalité t-SNE, UMAP !")
    while choiceExo != 0:
        choiceExo = int(input("Quelle exercice voulez-vous lancer ? :\n1. Exercice 1\n2. Exercice 2\n0. Quitter\n"))
        if choiceExo == 1:
            Exercice_1()
        elif choiceExo == 2:
            Exercice_2()
        elif choiceExo < 0 or choiceExo > 1:
            print("Choix invalide, veuillez réessayer.")