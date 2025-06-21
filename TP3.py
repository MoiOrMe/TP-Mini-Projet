from sklearn.datasets import make_swiss_roll, make_moons, make_classification
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Partie_1():
    # Swiss Roll
    X_swiss, color_swiss = make_swiss_roll(n_samples=1000, noise=0.05)

    # Moons
    X_moons, y_moons = make_moons(n_samples=500, noise=0.1)

    # Données linéaires simulées
    X_linear, y_linear = make_classification(n_samples=500, n_features=5, n_informative=2, n_redundant=0, n_clusters_per_class=1)

    # Swiss Roll en 3D
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2], c=color_swiss, cmap='Spectral')
    plt.title('Swiss Roll')
    plt.show()

    # Moons en 2D
    plt.figure(figsize=(5, 5))
    plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='coolwarm')
    plt.title('Données Moons')
    plt.axis('equal')
    plt.show()

    # Données linéaires en 2D
    plt.figure(figsize=(5, 5))
    plt.scatter(X_linear[:, 0], X_linear[:, 1], c=y_linear, cmap='viridis')
    plt.title('Données linéaires (features 0 et 1)')
    plt.axis('equal')
    plt.show()



def ChoixExoTP3():
    choiceExo = -1
    print("Bienvenue dans le TP de Méthodes de réduction de dimensionnalité t-SNE, UMAP !")
    while choiceExo != 0:
        choiceExo = int(input("Quelle partie voulez-vous lancer ? :\n1. Partie 1 (Work in progress)\n0. Quitter\n"))
        if choiceExo == 1:
            Partie_1()
        elif choiceExo < 0 or choiceExo > 1:
            print("Choix invalide, veuillez réessayer.")