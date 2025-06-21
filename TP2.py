def Exercice_1():
    print("Exercice : Work in progress")
    # Placeholder for future implementation
    pass

def ChoixExoTP2():
    choiceExo = -1
    print("Bienvenue dans le TP de Reconnaissance de Visages !")
    while choiceExo != 0:
        choiceExo = int(input("Quelle exercice voulez-vous lancer ? :\n1. Exercice (Work in progress)\n0. Quitter\n"))
        if choiceExo == 1:
            Exercice_1()
        elif choiceExo < 0 or choiceExo > 1:
            print("Choix invalide, veuillez réessayer.")