def GenerationDonnee():
    print("Génération et Visualisation du Jeu de Données")


def Exercice_1():
    print("KNN : K-Nearest Neighbors")


def Exercice_2():
    print("Arbre de Décision")


def Exercice_3():
    print("Random Forest")


def Exercice_4():
    print("Analyse comparative : Biais / Variance")


def ChoixExoTP5():
    choiceExo = -1
    print("Bienvenue dans le TP de Méthodes de réduction de dimensionnalité t-SNE, UMAP !")
    while choiceExo != 0:
        choiceExo = int(input("Quelle exercice voulez-vous lancer ? :\n1. Exercice 1 - KNN\n2. Exercice 2 - Arbre de Décision\n3. Exercice 3 - Random Forest\n4. Exercice 4 - Analyse comparative\n0. Quitter\n"))
        if choiceExo == 1:
            Exercice_1()
        elif choiceExo == 2:
            Exercice_2()
        elif choiceExo == 3:
            Exercice_3()
        elif choiceExo == 4:
            Exercice_4()
        elif choiceExo < 0 or choiceExo > 4:
            print("Choix invalide, veuillez réessayer.")