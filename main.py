from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def Exercice_1():
    print("Exercice 1 : Work in progress")
    # Placeholder for future implementation
    pass

def Exercice_2():
# Lire le dataset iris en utilisant la méthode datasets.load_iris() de sklearn.datasets. Explorer ce dataset.
    print("1.")

    iris = load_iris()
    X = iris.data
    y = iris.target

    print(pd.DataFrame(X, columns=iris.feature_names).head(), "\n")

# Appliquer l’ACP à ce dataset en utilisant 2 composantes principales. Vérifier la dimension des données après application de l’ACP.
    print("2.")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    print("Dimension de données : ", X_pca.shape, "\n")

# Afficher les composantes principales en utilisant principal.components_
    print("3.")
    print("Composantes principales :\n", pca.components_, "\n")

# Afficher les données dans le nouveau repère des composantes principales en utilisant 2 composantes. Afficher chaque classe avec une couleur différente.
    print("4.")

    plt.figure(figsize=(8,6))
    for target in set(y):
        plt.scatter(X_pca[y == target, 0], X_pca[y == target, 1], label=iris.target_names[target])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.title("Projection PCA (2D)")
    plt.show()

    print("Affichage sur matplotlib terminé.\n")

# Calculer et afficher la proportion de variance expliquée (explained_variance_ratio_) pour chaque composante principale.
    print("5.")

    print(pca.explained_variance_ratio_)
    print("Variance totale expliquée :", sum(pca.explained_variance_ratio_), "\n")

# Écrire un code Python permettant de calculer le nombre de dimensions requises pour préserver 95% de la variance du jeu de données :
# a. Utiliser la méthode cumsum() de numpy.
    print("6.")
    print("a.")

    pca_full = PCA().fit(X)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum >= 0.95) + 1
    print("Nombre de composantes pour ≥95% variance :", n_components_95, "\n")

# b. Proposer une autre méthode en jouant sur la valeur du paramètre n_components de la classe PCA de Scikit-learn. 
    print("b.")

    pca_95 = PCA(n_components=0.95)
    X_reduced = pca_95.fit_transform(X)
    print("Dimensions après réduction :", X_reduced.shape[1], "\n")

# Représenter la contribution de la variance en fonction du nombre de dimensions (représenter graphiquement cumsum). Interpréter.
    print("7.")

    plt.figure()
    plt.plot(np.arange(1, len(cumsum)+1), cumsum, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Variance expliquée cumulée")
    plt.title("Variance expliquée en fonction des dimensions")
    plt.grid(True)
    plt.show()

    print("Affichage sur matplotlib terminé.\n")

def Exercice_3():
    print("Exercice 3 : Work in progress")
    # Placeholder for future implementation
    pass

def Exercice_4():
    print("Exercice 4 : Work in progress")
    # Placeholder for future implementation
    pass

def main():
    choiceExo = -1
    print("Bienvenue dans le TP de PCA !")
    while choiceExo != 0:
        choiceExo = int(input("Quelle exercice voulez-vous lancer ? :\n1. Exercice 1 (Work in progress)\n2. Exercice 2\n3. Exercice 3 (Work in progress)\n4. Exercice 4 (Work in progress)\n0. Quitter\n"))
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

if __name__ == "__main__":
    main()