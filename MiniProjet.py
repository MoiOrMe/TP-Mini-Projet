import numpy as np
import matplotlib.pyplot as plt
import cv2


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

    dataset = [
        {'image': 'image1.jpg', 'gt_box': [50, 60, 100, 120]},
        {'image': 'image2.jpg', 'gt_box': [30, 40, 80, 90]},
        # ajouter d'autres images de ton dataset
    ]

    ious = []

    for data in dataset:
        img = cv2.imread(data['image'])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gt_box = data['gt_box']

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            # on prend la première prédiction comme exemple
            pred_box = faces[0]
            iou = compute_iou(gt_box, pred_box)
            ious.append(iou)
            print(f"Image: {data['image']} | IoU: {iou:.2f}")
        else:
            print(f"Image: {data['image']} | Aucun visage détecté.")
            ious.append(0)

    if ious:
        mean_iou = np.mean(ious)
        print(f"Mean IoU sur le dataset : {mean_iou:.2f}")


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