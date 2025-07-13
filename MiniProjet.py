import numpy as np
import pandas as pd
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog
from skimage import color, io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2
import os


def calculer_HOG(image_path):
    image = io.imread(image_path)
    image = image / 255.0
    
    if len(image.shape) == 3:
        image = color.rgb2gray(image)

    fd, hog_image = hog(image, orientations=12, pixels_per_cell=(4,4),
                        cells_per_block=(2,2), block_norm='L2-Hys',
                        visualize=True)

    plt.subplot(1,2,2)
    plt.imshow(hog_image, cmap='gray')
    plt.title('Image HOG')
    plt.axis('off')

    plt.show()

    print(f"Vecteur de caractéristiques HOG calculé. Taille : {fd.shape}")
    return fd


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    unionArea = boxAArea + boxBArea - interArea
    
    iou = interArea / unionArea if unionArea != 0 else 0
    return iou


def Partie_1():
    # Ecrire un programme Python permettant de détecter un visage à partir d’une image et d’un WebCam. Utiliser Cascade de Haar.
    print("Ecrire un programme Python permettant de détecter un visage à partir d'une image.")

    img = cv2.imread('visage.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Et d'une WebCam. Appuyer sur q to quit.")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow('Webcam Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Evaluer les performances de cette approche de détection sur un dataset de détection de visage public. Vous pouver utiliser le dataset suivant Kaggle Face Detection Dataset. (Obligatoire - Utiliser au moins la métrique IoU)
    print("Évaluation des performances de la détection de visage sur un dataset public.")

    images_folder = "kaggle/images/val"
    images_folder2 = "kaggle/images/train"
    labels2_folder = "kaggle/labels2"

    ious = []
    for label_file in os.listdir(labels2_folder):
        if not label_file.endswith(".txt"):
            continue

        image_name = label_file.replace(".txt", ".jpg")
        image_path = os.path.join(images_folder, image_name)
        if image_path is None or not os.path.exists(image_path):
            image_path = os.path.join(images_folder2, image_name)
        label_path = os.path.join(labels2_folder, label_file)
        print(f"Traitement de l'image : {image_name}")

        img = cv2.imread(image_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gt_boxes = []
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 6 and parts[0] == "Human":
                    x_min = float(parts[2])
                    y_min = float(parts[3])
                    x_max = float(parts[4])
                    y_max = float(parts[5])
                    w = x_max - x_min
                    h = y_max - y_min
                    gt_boxes.append([x_min, y_min, w, h])

        print("Lecture des boîtes englobantes GT :", gt_boxes)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
        
        for gt_box in gt_boxes:
            max_iou = 0
            for pred_box in faces:
                iou = compute_iou(gt_box, pred_box)
                max_iou = max(max_iou, iou)
            ious.append(max_iou)
            print(f"{image_name} | IoU: {max_iou:.2f}")

    if ious:
        mean_iou = np.mean(ious)
        print(f"Mean IoU sur le dataset : {mean_iou:.2f}")

    print("Interpréter les résultats obtenus.\n" \
          "Dans un premier test, la moyenne des IoU était de 0.27. Donc on peut en déduire que la précison de notre modèle Haar n'était pas encore optimal et qu'il fallait modifier et tester de nouveaux paramètres. Le scaleFactor à 1.1 et le minNeighbors à 7\n" \
          "Dans le suivant, on à réussi à pervenir à une moyenne des IoU du dataset à 0.35. Le scaleFactor à 1.05 et le minNeighbors à 3." )


def Partie_2():
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names
    results = []
    
    print("\n=== PCA + KNN ===")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    n_components = 150
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pca, y_train)

    y_pred = knn.predict(X_test_pca)

    acc_pca_knn = accuracy_score(y_test, y_pred)
    results.append(['PCA + KNN', acc_pca_knn])

    print("Rapport de classification :")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    plt.imshow(X_test[0].reshape(lfw_people.images.shape[1:]), cmap='gray')
    plt.title(f"Vérité : {target_names[y_test[0]]} | Prédit : {target_names[y_pred[0]]}")
    plt.axis('off')
    plt.show()

    print("\n=== HOG + Decision Tree ===")
    hog_features_list = []
    for img in lfw_people.images:
        image = img / 255.0
        fd = hog(image, orientations=12, pixels_per_cell=(4,4),
                cells_per_block=(2,2), block_norm='L2-Hys')
        hog_features_list.append(fd)

    X_hog = np.array(hog_features_list)
    
    X_train_hog, X_test_hog, y_train_hog, y_test_hog = train_test_split(X_hog, y, test_size=0.3, random_state=42, stratify=y)

    dt = DecisionTreeClassifier(max_depth=30, random_state=42)
    dt.fit(X_train_hog, y_train_hog)

    y_pred_dt = dt.predict(X_test_hog)

    acc_dt = accuracy_score(y_test_hog, y_pred_dt)
    results.append(['HOG + Decision Tree', acc_dt])

    print("Rapport de classification (Decision Tree) :")
    print(classification_report(y_test_hog, y_pred_dt, target_names=target_names, zero_division=0))

    print("\n=== HOG + Random Forest ===")
    rf = RandomForestClassifier(n_estimators=200, max_depth=30, max_features='sqrt', random_state=42)
    rf.fit(X_train_hog, y_train_hog)

    y_pred_rf = rf.predict(X_test_hog)

    acc_rf = accuracy_score(y_test_hog, y_pred_rf)
    results.append(['HOG + Random Forest', acc_rf])

    print("Rapport de classification (Random Forest) :")
    print(classification_report(y_test_hog, y_pred_rf, target_names=target_names, zero_division=0))

    print("\n=== CNN ===")
    X_cnn = lfw_people.images / 255.0
    y_cnn = to_categorical(y)

    X_cnn = X_cnn.reshape((-1, X_cnn.shape[1], X_cnn.shape[2], 1))

    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y_cnn, test_size=0.3, random_state=42, stratify=y)

    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=X_train_cnn.shape[1:]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(y_cnn.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train_cnn, y_train_cnn, epochs=50, batch_size=32, validation_split=0.2)

    loss, acc_cnn = model.evaluate(X_test_cnn, y_test_cnn)
    results.append(['CNN', acc_cnn])

    print("\n=== Tableau récapitulatif des performances ===")
    df_results = pd.DataFrame(results, columns=['Méthode', 'Accuracy'])
    print(df_results.to_string(index=False))

    print("\nConsigne :\nInterpréter les résultats obtenus.")


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