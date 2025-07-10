import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


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

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
        
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
          "Pour l'insatnt , la moyenne des IoU est de {:.2f}. Donc on peut en déduire que la précison de notre modèle Haar n'est pas encore optimal, faudrais modifié et tester de nouveaux paramètres afin d'optimiser la détection des visages." )


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