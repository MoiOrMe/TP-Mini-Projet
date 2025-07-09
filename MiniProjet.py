def Partie_1():
    print("Lancement de la Partie 1 du Mini-Projet : Détection et reconnaissance faciale avec apprentissage automatique traditionnel.")
    # Ici, vous pouvez ajouter le code spécifique pour la Partie 1


def Partie_2():
    print("Lancement de la Partie 2 du Mini-Projet : Détection et reconnaissance faciale avec apprentissage profond.")
    # Ici, vous pouvez ajouter le code spécifique pour la Partie 2


def ChoixMiniProjet() :
    choiceExo = -1
    print("Bienvenue dans le Mini-Projet de Détection et reconnaissance faciale !")
    while choiceExo != 0:
        choiceExo = int(input("Quelle partie voulez-vous lancer ? :\n1. Partie 1 - Détection de visages\n2. Partie 2 - Reconnaissance de visages\n0. Quitter\n"))
        if choiceExo == 1:
            Partie_1()
        elif choiceExo == 2:
            Partie_2()
        elif choiceExo < 0 or choiceExo > 2:
            print("Choix invalide, veuillez réessayer.")