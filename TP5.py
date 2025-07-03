import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    
    plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k')
    plt.title(title)
    plt.show()


def GenerationDonnee():
    print("Génération et Visualisation du Jeu de Données")

#    X_circles, y_circles = make_circles(n_samples=500, noise=0.2, factor=0.5, random_state=42)
    X_moons, y_moons = make_moons(n_samples=500, noise=0.2, random_state=42)

#   X_train, X_test, y_train, y_test = train_test_split(X_circles, y_circles, test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons, test_size=0.3, random_state=42)

    plt.figure(figsize=(6,4))
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='coolwarm', label='Train')
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='coolwarm', marker='x', label='Test')
    plt.title('Jeu de données : train (points) et test (croix)')
    plt.legend()
    plt.show()

# Questions de compréhension
    print("Questions de compréhension :\n"
          "Pourquoi séparer les données ?\n"
          "Pour entraîner le modèle sur certaine donnée et garder les autres pour le tester, sinon ça fais que mémoriser les données et non pas classer.\n" \
          "\nQuelques définitions :\n"
          "Frontières de décision : Ce sont les zones séparant les classes dans l'espace des features selon le modèle.\n"
          "Accuracy : C'est la proportion de bonnes prédictions.\n"
          "Precision : C'est la proportion de vrais positifs parmi les prédits positifs.\n"
          "Recall : C'est la proportion de vrais positifs parmi les réels positifs.\n"
          "F1-score : C'est la moyenne harmonique entre la précision et le rappel.\n"
          "Validation croisée : Divise le train en k folds pour entraîner sur k-1 et valider sur le fold restant, afin d'estimer la performance moyenne et détecter la variance.\n")
    
    return X_train, X_test, y_train, y_test


def Exercice_1():
    print("KNN : K-Nearest Neighbors")

    X_train, X_test, y_train, y_test = GenerationDonnee()

    k_values = [1, 3, 7, 15]

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        y_pred = knn.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n=== KNN (k={k}) ===")
        print(f"Accuracy : {acc:.2f}")
        print(f"Precision : {prec:.2f}")
        print(f"Recall : {rec:.2f}")
        print(f"F1-score : {f1:.2f}")
        
        cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='f1')
        print(f"F1-score (Validation croisée) : {cv_scores.mean():.2f} +/- {cv_scores.std():.2f} \n")

        plot_decision_boundary(knn, X_train, y_train, f"KNN (k={k}) - Train")
    
    print("Interpréter le biais et la variance :\n" 
          "Plus le biais est élevé, plus le modèle est en sous-apprentissage. C'est àa dire que le modèle est trop simple.\n"
          "Plus la variance est élevée, plus le modèle est en sur-apprentissage. C'est à dire que le modèle est trop complexe et s'adapte trop aux données d'entraînement.\n")


def Exercice_2():
    print("Arbre de Décision")

    X_train, X_test, y_train, y_test = GenerationDonnee()

    depth_values = [2, 5, 10]

    for depth in depth_values:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
                
        y_pred = tree.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n=== Arbre de Décision (max_depth={depth}) ===")
        print(f"Accuracy : {acc:.2f}")
        print(f"Precision : {prec:.2f}")
        print(f"Recall : {rec:.2f}")
        print(f"F1-score : {f1:.2f} \n")

        cv_scores = cross_val_score(tree, X_train, y_train, cv=5, scoring='f1')
        print(f"F1-score (Validation croisée) : {cv_scores.mean():.2f} +/- {cv_scores.std():.2f} \n")

        plot_decision_boundary(tree, X_train, y_train, f"Arbre de Décision (max_depth={depth}) - Train")

    print("Observer le comportement du modèle dans le cas d'un arbre trop profond et peu profond.\n"
          "Un arbre trop profond va sur-apprendre les données d'entraînement, ce qui peut mener à une variance élevée et un sur-apprentissage.\n"
          "Un arbre peu profond va sous-apprendre les données, ce qui peut mener à un biais élevé et un sous-apprentissage.\n")


def Exercice_3():
    print("Random Forest")

    X_train, X_test, y_train, y_test = GenerationDonnee()

    n_estimators_values = [10, 100]
    max_depth_values = [5, 10]

    for n in n_estimators_values:
        for depth in max_depth_values:
            rf = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=42)
            rf.fit(X_train, y_train)
                        
            y_pred = rf.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            print(f"=== Random Forest (n_estimators={n}, max_depth={depth}) ===")
            print(f"Accuracy : {acc:.2f}")
            print(f"Precision : {prec:.2f}")
            print(f"Recall : {rec:.2f}")
            print(f"F1-score : {f1:.2f}")

            cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')
            print(f"F1-score (Validation croisée) : {cv_scores.mean():.2f} +/- {cv_scores.std():.2f} \n")
            
            plot_decision_boundary(rf, X_train, y_train, f"Random Forest (n_estimators={n}, max_depth={depth}) - Train")

    print("Comparer les scores aux précédents modèles :\n"
          "\n"
          "Discuter de la stabilité et de la robustesse :\n"
          "\n"
          "Montrer l'effet d'agrégation des arbres :\n"
          "\n")


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