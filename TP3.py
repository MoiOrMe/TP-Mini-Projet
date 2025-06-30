from sklearn.datasets import make_swiss_roll, make_moons, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
os.environ["OMP_NUM_THREADS"] = "1"


def Exercice_1():
    # Partie 1 – Génération et visualisation des données
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

    # Normalisation des données
    X_swiss_scaled = StandardScaler().fit_transform(X_swiss)
    X_moons_scaled = StandardScaler().fit_transform(X_moons)
    X_linear_scaled = StandardScaler().fit_transform(X_linear)


    # Partie 2 – Application des méthodes
    # PCA sur Swiss Roll
    pca_swiss = PCA(n_components=2)
    X_swiss_pca = pca_swiss.fit_transform(X_swiss_scaled)

    print("Variance expliquée (Swiss Roll) :", pca_swiss.explained_variance_ratio_.sum())

    plt.figure(figsize=(5, 5))
    plt.scatter(X_swiss_pca[:, 0], X_swiss_pca[:, 1], c=color_swiss, cmap='Spectral')
    plt.title('PCA - Swiss Roll')
    plt.axis('equal')
    plt.show()

    # PCA sur Moons
    pca_moons = PCA(n_components=2)
    X_moons_pca = pca_moons.fit_transform(X_moons_scaled)

    print("Variance expliquée (Moons) :", pca_moons.explained_variance_ratio_.sum())

    plt.figure(figsize=(5, 5))
    plt.scatter(X_moons_pca[:, 0], X_moons_pca[:, 1], c=y_moons, cmap='coolwarm')
    plt.title('PCA - Moons')
    plt.axis('equal')
    plt.show()

    # PCA sur Données linéaires
    pca_linear = PCA(n_components=2)
    X_linear_pca = pca_linear.fit_transform(X_linear_scaled)

    print("Variance expliquée (linéaire) :", pca_linear.explained_variance_ratio_.sum(), "\n")

    plt.figure(figsize=(5, 5))
    plt.scatter(X_linear_pca[:, 0], X_linear_pca[:, 1], c=y_linear, cmap='viridis')
    plt.title('PCA - Données linéaires')
    plt.axis('equal')
    plt.show()

    # Commentez la capacité de PCA à préserver la structure des données
    print("Commentez la capacité de PCA à préserver la structure des données :\n"
          "Swiss Roll : PCA ne préserve pas la structure en spirale, car il projette les données dans un espace de dimension inférieure sans tenir compte de la topologie.\n"
          "Moons : PCA peut capturer la structure globale, mais ne préserve pas les détails locaux des deux lunes.\n"
          "Données linéaires : PCA fonctionne bien pour les données linéaires, car elles sont déjà dans un espace de faible dimension.\n")
    
    # t-SNE sur Swiss Roll
    tsne_swiss = TSNE(n_components=2, perplexity=30, random_state=42)
    X_swiss_tsne = tsne_swiss.fit_transform(X_swiss_scaled)

    plt.figure(figsize=(5, 5))
    plt.scatter(X_swiss_tsne[:, 0], X_swiss_tsne[:, 1], c=color_swiss, cmap='Spectral')
    plt.title('t-SNE - Swiss Roll')
    plt.axis('equal')
    plt.show()

    # t-SNE sur Moons
    tsne_moons = TSNE(n_components=2, perplexity=30, random_state=42)
    X_moons_tsne = tsne_moons.fit_transform(X_moons_scaled)

    plt.figure(figsize=(5, 5))
    plt.scatter(X_moons_tsne[:, 0], X_moons_tsne[:, 1], c=y_moons, cmap='coolwarm')
    plt.title('t-SNE - Moons')
    plt.axis('equal')
    plt.show()

    # t-SNE sur Données linéaires
    tsne_linear = TSNE(n_components=2, perplexity=30, random_state=42)
    X_linear_tsne = tsne_linear.fit_transform(X_linear_scaled)

    plt.figure(figsize=(5, 5))
    plt.scatter(X_linear_tsne[:, 0], X_linear_tsne[:, 1], c=y_linear, cmap='viridis')
    plt.title('t-SNE - Données linéaires')
    plt.axis('equal')
    plt.show()

    # Discutez de l’effet du paramètre perplexity.
    print("Discutez de l’effet du paramètre perplexity :\n"
          "Le paramètre perplexity dans t-SNE contrôle le compromis entre la préservation des distances locales et globales. "
          "Qu'est-ce que ça veux dire ? Eh bien plus la perplexité est faible plus elle met l'accent sur les relations locales, tandis qu'une grande perplexité capture mieux la structure globale. "
          "Etre trop concentré sur la structure nous fais perdre les détails locaux tandis que lorsqu'on se focus sur les relation local, ça risque de créer plein de cluster\n")
    
    # Observez si les clusters sont bien séparés.
    print("Observez si les clusters sont bien séparés :\n"
          "Lorsque la perplexity est à 5, les cluster sont ultra séparé, tandis que si au contraire on passe à 50, il y a moins de cluster.\n"
          "C'est pour ça qu'il est importznt de choisir un perplexity ni trop grand ni trop petit par rapport à la structure des données.\n")
    
    # UMAP sur Swiss Roll
    umap_swiss = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X_swiss_umap = umap_swiss.fit_transform(X_swiss_scaled)

    plt.figure(figsize=(5, 5))
    plt.scatter(X_swiss_umap[:, 0], X_swiss_umap[:, 1], c=color_swiss, cmap='Spectral')
    plt.title('UMAP - Swiss Roll')
    plt.axis('equal')
    plt.show()

    # UMAP sur Moons
    umap_moons = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X_moons_umap = umap_moons.fit_transform(X_moons_scaled)

    plt.figure(figsize=(5, 5))
    plt.scatter(X_moons_umap[:, 0], X_moons_umap[:, 1], c=y_moons, cmap='coolwarm')
    plt.title('UMAP - Moons')
    plt.axis('equal')
    plt.show()

    # UMAP sur Données linéaires
    umap_linear = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X_linear_umap = umap_linear.fit_transform(X_linear_scaled)

    plt.figure(figsize=(5, 5))
    plt.scatter(X_linear_umap[:, 0], X_linear_umap[:, 1], c=y_linear, cmap='viridis')
    plt.title('UMAP - Données linéaires')
    plt.axis('equal')
    plt.show()

    # Comparez les résultats avec t-SNE
    print("Comparez les résultats avec t-SNE :\n"
          "UMAP est bien plus rapide et conserve aussi mieux la structure de base des données par rapport à t-SNE, du moins dans la plupart des cas. Pour les données non linéaires comme Swiss Roll et Moons, UMAP capture mieux la structure et les relations locales que t-SNE.\n"
          "\nCommentez la forme et la fidélité de la représentation.\n"
          "Pour les lunes, t-SNE nous montre plus facilement les croissants de base tandis que UMAP nous montre les deux lunes bien séparées. Pour le Swiss Roll, UMAP préserve mieux la forme en spirale que t-SNE. On peut le voir grâce aux schéma précédent.\n")
    
    # Partie 3 – Questions de compréhension
    # Quelle méthode vous semble la plus adaptée pour la visualisation ?
    print("Quelle méthode vous semble la plus adaptée pour la visualisation ?\n"
          "UMAP tout les jours en termes de technicité. Elle fais le même travaille que t-SNE dans certaines condition en meilleur et en même temps en prenant moins de temps. Le seul défaut que j'ai c'est qu'il soit pas dans sklearn.\n")
    
    # Laquelle privilégieriez-vous pour réduire la dimension avant un modèle supervisé ?
    print("Laquelle privilégieriez-vous pour réduire la dimension avant un modèle supervisé ?\n"
          "Pour les données linéaires, PCA car elle garde la variance et fais un très bon rendu. Pour les données non linéaires, je préfèrerais UMAP. Car comme dit précédément, elle conserve mieux la structure des données et est plus rapide que t-SNE.\n")
    
    # Comment choisir entre t-SNE et UMAP dans une tâche réelle ?
    print("Comment choisir entre t-SNE et UMAP dans une tâche réelle ?\n"
          "UMAP sera en général choici pour sa rapidité et sa capacité à préserver la structure des données. Mais, t-SNE peut être utile dans certains cas où la séparation des clusters est cruciale lorsq de la visualisation. Enfaite le choix dépendra de notre objectifs avec les données.\n")
    
    # En quoi la linéarité d’une méthode influence-t-elle ses résultats ?
    print("En quoi la linéarité d’une méthode influence-t-elle ses résultats ?\n"
          "PCA ne peux pas capturer la non-linéarité d'une structure, tandis que pour t-SNE et UMAP ils le peuvent. Donc ça change les résultats de tout au tout.\n")


def ChoixExoTP3():
    choiceExo = -1
    print("Bienvenue dans le TP de Méthodes de réduction de dimensionnalité t-SNE, UMAP !")
    while choiceExo != 0:
        choiceExo = int(input("Quelle exercice voulez-vous lancer ? :\n1. Exercice\n0. Quitter\n"))
        if choiceExo == 1:
            Exercice_1()
        elif choiceExo < 0 or choiceExo > 1:
            print("Choix invalide, veuillez réessayer.")