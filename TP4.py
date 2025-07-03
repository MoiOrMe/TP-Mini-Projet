import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage


def Exercice_1():
    print("Exercice 1 : Segmentation d'images par K-Means")

# Choisir une image couleur de votre choix. Lire cette image et l’afficher. Pour cela vous pouvez utiliser les librairies OpenCV sous python (CV2). Pour l’affichage vous pouvez utiliser matplotlib. 
    image = cv2.imread('voiture_rouge.jpg')

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.title("Voiture Rouge")
    plt.axis('off')
    plt.show()

# Prétraiter l’image : Aplatissez les dimensions de l’image pour créer un ensemble de données constitué de pixels, chacun représenté par ses trois valeurs RGB.
    h, w, c = image_rgb.shape
    pixels = image_rgb.reshape((-1, 3))

    print("Forme d'origine :", image_rgb.shape)
    print("Forme après reshape :", pixels.shape)

# Appliquer le clustering k-means : Utilisez l’algorithme de clustering k-means pour  regrouper les pixels en fonction de leurs valeurs RGB. Utiliser l’implémentation Kmeans de scikit-learn.
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    print("Centroids shape:", centroids.shape)
    print("Labels shape:", labels.shape)

# Générer l’image segmentée : Attribuez chaque pixel à son centroïde de cluster correspondant pour créer une image segmentée.
    segmented_pixels = centroids[labels]
    segmented_image = segmented_pixels.reshape(image_rgb.shape)
    segmented_image = segmented_image.astype(np.uint8)

# Visualiser l’image segmentée : Affichez l’image segmentée afin d’observer le résultat de la segmentation.
    plt.imshow(segmented_image)
    plt.title(f"Image segmentée (k={k})")
    plt.axis('off')
    plt.show()

# Ajuster les paramètres et analyser : Expérimentez avec différentes valeurs de k pour atteindre le niveau de segmentation souhaité et analysez les résultats obtenus.
    print("Essayez différentes valeurs de k pour voir comment la segmentation change.")
    for k in [2, 3, 4, 6]:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pixels)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        segmented_pixels = centroids[labels]
        segmented_image = segmented_pixels.reshape(image_rgb.shape)
        segmented_image = segmented_image.astype(np.uint8)

        plt.imshow(segmented_image)
        plt.title(f"Image segmentée (k={k})")
        plt.axis('off')
        plt.show()


def Exercice_2():
# Exercice pratique
    digits = load_digits()
    data = digits.data
    labels = digits.target

    print("Shape des données :", data.shape)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    print("Shape après PCA :", data_pca.shape)

    Z = linkage(data_pca, method='ward')

    plt.figure(figsize=(10, 7))
    dendrogram(Z, truncate_mode='level', p=20)
    plt.title("Dendrogramme clustering hiérarchique (ward)")
    plt.xlabel("Index des points")
    plt.ylabel("Distance")
    plt.show()
# Question
    print("1. Quel est le rôle de la distance euclidienne dans ces méthodes ?\n"
          "\n"
          "2. En quoi la PCA facilite-t-elle le clustering visuel ?\n"
          "\n"
          "3. Quels sont les cas d'usage typiques du clustering dans les systèmes de vision par ordinateur ?\n"
          "\n"
          "4. Peut-on utiliser ces clusters comme pseudo-labels ? Pourquoi ?\n"
          "\n")


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