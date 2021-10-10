import cv2
import numpy as np
import os 
import pandas as pd
import torch

#Загружаем модель и веса
from model_structure import model
model.load_state_dict(torch.load('model_epoch30.pth'))
model.eval()

#Загружаем каскады Хаара, обученные детектировать лица
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Определяем значения параметров для вывода надписей
scaling_factor = 0.6
gender_dict = {0: "Пол: мужской", 1: "Пол: женский"}
etn_dict = {0: "Этническая группа: Европейская", 1: "Этническая группа: Африканская", 2: "Этническая группа: Азиаткская",
           3: "Этническая группа: Индийская", 4: "Этническая группа: Арабская"}
font = cv2.FONT_HERSHEY_COMPLEX
fontScale = 0.4

#Запускаем веб-камеру, првоодим детекцию и классификацию в каждом кадре и выдаём ответ
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

#детектированное лицо передаём в нейронную сеть и выводим полученные значения
    for (x,y,w,h) in face_rects:
        image_model = gray[y:y+h, x:x+w]
        image_model = cv2.resize(image_model, (48,48), interpolation = cv2.INTER_AREA)
        gender, age, etn = model(torch.tensor(image_model).unsqueeze(0).unsqueeze(0).float())

        etn = np.argmax(etn.detach().numpy())
        gender = int(np.round(gender.detach().numpy()))

        #рисуем прямоугольник и надписи
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        frame = cv2.putText(frame, gender_dict[gender],(x+w,y), font, fontScale,(255,255,255),1, cv2.LINE_AA)
        frame = cv2.putText(frame, etn_dict[etn], (x+w,y+25), font, fontScale,(255,255,255),1, cv2.LINE_AA)
        frame = cv2.putText(frame, f"Возраст: {int(torch.round(age))}",(x+w,y+50), font, fontScale,(255,255,255),1, cv2.LINE_AA)

    # выводим на экран
    cv2.imshow('Face Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()