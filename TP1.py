from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

    print("Avant normalisation :")
    print(pd.DataFrame(X, columns=iris.feature_names).head(), "\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Après normalisation :")
    print(pd.DataFrame(X_scaled, columns=iris.feature_names).head(), "\n")

# Appliquer l’ACP à ce dataset en utilisant 2 composantes principales. Vérifier la dimension des données après application de l’ACP.
    print("2.")

    pca = PCA(n_components=2)

    print("Sans normalisation :")
    X_pca = pca.fit_transform(X)
    print("Dimension de données : ", X_pca.shape, "\n")

    print("Avec normalisation :")
    X_pca_scaled = pca.fit_transform(X_scaled)
    print("Dimension de données : ", X_pca_scaled.shape, "\n")

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
    plt.title("Projection PCA (2D) sans normalisation")
    plt.show()

    plt.figure(figsize=(8,6))
    for target in set(y):
        plt.scatter(X_pca_scaled[y == target, 0], X_pca_scaled[y == target, 1], label=iris.target_names[target])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.title("Projection PCA (2D) avec normalisation")
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

    print("Sans normalisation :")
    pca_full = PCA().fit(X)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum >= 0.95) + 1
    print("Nombre de composantes pour ≥95% variance :", n_components_95, "\n")

    print("Avec normalisation :")
    pca_full_scaled = PCA().fit(X_scaled)
    cumsum_scaled = np.cumsum(pca_full_scaled.explained_variance_ratio_)
    n_components_95_scaled = np.argmax(cumsum_scaled >= 0.95) + 1
    print("Nombre de composantes pour ≥95% variance :", n_components_95_scaled, "\n")

# b. Proposer une autre méthode en jouant sur la valeur du paramètre n_components de la classe PCA de Scikit-learn. 
    print("b.")

    print("Sans normalisation :")
    pca_95 = PCA(n_components=0.95)
    X_reduced = pca_95.fit_transform(X)
    print("Dimensions après réduction :", X_reduced.shape[1], "\n")

    print("Avec normalisation :")
    pca_95 = PCA(n_components=0.95)
    X_reduced_scaled = pca_95.fit_transform(X_scaled)
    print("Dimensions après réduction :", X_reduced_scaled.shape[1], "\n")

# Représenter la contribution de la variance en fonction du nombre de dimensions (représenter graphiquement cumsum). Interpréter.
    print("7.")

    plt.figure()
    plt.plot(np.arange(1, len(cumsum)+1), cumsum, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Variance expliquée cumulée")
    plt.title("Variance expliquée en fonction des dimensions non normalisées")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(np.arange(1, len(cumsum_scaled)+1), cumsum_scaled, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Variance expliquée cumulée")
    plt.title("Variance expliquée en fonction des dimensions normalisées")
    plt.grid(True)
    plt.show()

    print("Affichage sur matplotlib terminé.\n")

def Exercice_3():
# Appliquez l’ACP aux données et affichez la décroissance des valeurs propres.
    print("1.")

    df = pd.read_csv("leaf/leaf.csv")
    print("Shape du dataset : ", df.shape)

    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(pca.explained_variance_)+1), pca.explained_variance_, marker='o')
    plt.xlabel("Composante principale")
    plt.ylabel("Valeur propre")
    plt.title("Décroissance des valeurs propres")
    plt.grid(True)
    plt.show()

    print("Affichage sur matplotlib terminé.\n")

# Affichez les projections des données sur les 3 premiers axes principaux en utilisant l’étiquette de classe pour donner une couleur à chaque classe.
    print("2.")

    pca_3 = PCA(n_components=3)
    X_3d = pca_3.fit_transform(X_scaled)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y, cmap='tab20', s=40)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title("Projection 3D des feuilles (ACP)")
    plt.colorbar(scatter, label="Classe (espèce)")
    plt.show()

    print("Affichage sur matplotlib terminé.\n")

def Exercice_4():
    print("Exercice 4 : Work in progress")
    # Placeholder for future implementation
    pass

def ChoixExoTP1():
    choiceExo = -1
    print("Bienvenue dans le TP de PCA !")
    while choiceExo != 0:
        choiceExo = int(input("Quelle exercice voulez-vous lancer ? :\n1. Exercice 1 (Work in progress)\n2. Exercice 2\n3. Exercice 3\n4. Exercice 4 (Work in progress)\n0. Quitter\n"))
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