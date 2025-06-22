from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


def plot_gallery(images, titles, h, w, n_row=3, n_col=5):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def Exercice():
    # Exercice 1 - Lire le dataset « fetch_lfw_people ».
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    images = lfw_people.images
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names

    n_samples, h, w = images.shape
    n_features = X.shape[1]
    n_classes = target_names.shape[0]

    print(f"\nNombre d'images : {n_samples}")
    print(f"Dimension des images : {h} x {w}")
    print(f"Nombre de classes : {n_classes}\n")

    # Exercice 2 - Afficher quelques images comme suit la figure ci-contre.
    titles = [target_names[y[i]] for i in range(15)]

    plot_gallery(images, titles, h, w)
    plt.suptitle("Premières images du dataset", fontsize=16)
    plt.show()

    # Exercice 3 - Afficher le nombre d’images par personne sous forme d’histogramme. Que constatez-vous ?
    counts = np.bincount(y)

    plt.figure(figsize=(10, 5))
    plt.bar(target_names, counts, color='skyblue')
    plt.title("Nombre d'images par personne")
    plt.xlabel("Personne")
    plt.ylabel("Nombre d'images")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    print("Que constatez-vous ? :\n"
          "On constate graâce à l'histogramme que chaque personne n'a pas le même nombre d'images.\n")
    
    # Exercice 4 - Appliquer PCA pour réduire la dimension des données.
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Application de PCA
    n_components = 150
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=False)
    X_pca = pca.fit_transform(X_scaled)

    print(f"Dimension d'origine : {X.shape[1]}")
    print(f"Dimension réduite : {X_pca.shape[1]}")

    plt.figure(figsize=(8, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Variance expliquée cumulée")
    plt.title("Variance expliquée par la PCA")
    plt.grid(True)
    plt.axhline(0.95, color='r', linestyle='--', label='95%')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Exercice 5 - Appliquer PCA avec l’option whiten =True. Expliquer l’intérêt de cette option pour cette application
    pca_white = PCA(n_components=150, svd_solver='randomized', whiten=True)
    X_pca_white = pca_white.fit_transform(X_scaled)

    print("Shape après PCA avec whiten:", X_pca_white.shape)

    plt.figure(figsize=(10, 4))
    plt.plot(pca.explained_variance_, label='Sans whiten')
    plt.plot(pca_white.explained_variance_, label='Avec whiten')
    plt.title("Comparaison des variances des composantes")
    plt.xlabel("Composante")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Expliquer l'intérêt de cette option pour cette application :\n"
          "Alors qu'est-ce que ça fait, ça divise chaque composante principale par son écart-type et ça rend les composantes non corrélées et de variance unitaire."
          "C'est utile car ainsi, toute les dimension ont la même importance et ça évite que les premières composantes impact plus les résultats que les autres.\n")
    
    # Exercice 6 - On vous propose de rééquilibrer le jeu de données en se limitant à 50 images par personne. Écrire le code correspondant.
    class_counts = defaultdict(int)

    X_balanced = []
    y_balanced = []

    for xi, yi in zip(X, y):
        if class_counts[yi] < 50:
            X_balanced.append(xi)
            y_balanced.append(yi)
            class_counts[yi] += 1

    X_balanced = np.array(X_balanced)
    y_balanced = np.array(y_balanced)

    print(f"Nouvelle taille du dataset : {X_balanced.shape}")

    counts_balanced = np.bincount(y_balanced)
    plt.figure(figsize=(10, 5))
    plt.bar(target_names, counts_balanced, color='salmon')
    plt.title("Dataset rééquilibré : max 50 images/personne")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Nombre d'images")
    plt.tight_layout()
    plt.show()

    # Exercice 7 - Interpréter
    print("Interprétation :\n"
          "Dans le datasets initial, les données n'étaient pas équilibré. Cela ne rendais pas optimal la préparation de données pour un apprentissage. On a donc alors réequilibre à 50 images par classe.\n"
          "De plus, le dataset avais une dimension trop élevé. On a donc dû choisir de les redimensionner, onperds un peu de l'information mais on en grade l'essentiel. ET ça nous permet d'avoir un gain de temps important et ça évite l'overfitting par la même occasion.\n"
          "On a utilisé le whiten dans l'exercice 5, ça nous permet de rendre chaque composante non corrélée et de variance unitaire. C'est est utile pour éviter que certaines composantes n'impactent plus les résultats que d'autres.\n"
          "DONC, on a commencé par prétraiter les données, en les normalisant et en les réduisant avec PCA. Ensuite, on a rééquilibré le dataset pour avoir un nombre d'images constant par classe. Et ainsi, on a un dataset propre et préparé pour de bonne perf en reconnaissance d'image, si j'ai bien compris le truc.\n")
    

def ChoixExoTP2():
    choiceExo = -1
    print("Bienvenue dans le TP de Reconnaissance de Visages !")
    while choiceExo != 0:
        choiceExo = int(input("Quelle exercice voulez-vous lancer ? :\n1. Exercice\n0. Quitter\n"))
        if choiceExo == 1:
            Exercice()
        elif choiceExo < 0 or choiceExo > 1:
            print("Choix invalide, veuillez réessayer.")