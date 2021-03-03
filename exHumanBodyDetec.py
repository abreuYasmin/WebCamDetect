import cv2
import sys

cascPath = sys.argv[0]
fullCascade = cv2.CascadeClassifier('haarcascades/haarcascade_fullbody.xml')
upperCascade = cv2.CascadeClassifier('haarcascades/haarcascade_upperbody.xml')
lowerCascade = cv2.CascadeClassifier('haarcascades/haarcascade_lowerbody.xml')

video_capture = cv2.VideoCapture(0)

# frame by frame
while True:
    ret, frame = video_capture.read()  # one frame by each loop
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fullBodies = fullCascade.detectMultiScale(
        gray,
        scaleFactor=2.5,
        minNeighbors=5,
        minSize=(15, 15),
        maxSize=(100, 100)
    )
    upperBodies = upperCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # how much the image size is reduced at each image scale
        minNeighbors=5,  # how many neighbors each candidate rectangle should have to retain it
        minSize=(15, 15),  # minimum object size, objects smaller are ignored
        maxSize=(50, 50)  # maximum object size, objects larger than that are ignored
    )
    lowerBodies = lowerCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(15, 15),
        maxSize=(50, 50)
    )

    # retângulo
    for (x, y, w, h) in fullBodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Entire body', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)
    for (x, y, w, h) in upperBodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, 'Upper body', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
    for (x, y, w, h) in lowerBodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Lower body', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)


    cv2.imshow('Detecção de corpos - pressione a tecla "Q" para sair', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # key == q (quit)
        break

video_capture.release()
cv2.destroyAllWindows()





