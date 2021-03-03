import cv2
import sys

cascPath = sys.argv[0]
smileCascade = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')
frontalCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()  # one frame by each loop
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frontalFaces = frontalCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=3,
        minSize=(25, 25),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    smiles = smileCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=60,
        minSize=(15, 15),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in frontalFaces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (35,142,35), 2)
        cv2.putText(frame, 'Frontal face', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (35,142,35), 2)
    for (x, y, w, h) in smiles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50,205,50), 2)
        cv2.putText(frame, 'Smile', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (50,205,50), 2)

    cv2.imshow('Detecção de sorrisos - pressione a tecla "Q" para sair', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # key == q (quit)
        break

video_capture.release()
cv2.destroyAllWindows()
