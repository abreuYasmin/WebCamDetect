import cv2
import sys

cascPath = sys.argv[0]
frontalCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
sideCascade = cv2.CascadeClassifier('haarcascades/haarcascade_profileface.xml')
profileCascade = cv2.CascadeClassifier('haarcascades/haarcascade_profileface.xml')


video_capture = cv2.VideoCapture(0)

# frame by frame
while True:
    ret, frame = video_capture.read()  # one frame by each loop
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frontalFaces = frontalCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=3,
        minSize=(35, 35),

    )
#     sideFaces = sideCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.5,  # how much the image size is reduced at each image scale
#         minNeighbors=3,  # how many neighbors each candidate rectangle should have to retain it
#         minSize=(25, 25),  # minimum object size, objects smaller are ignored
# #        maxSize=(50, 50)  # maximum object size, objects larger than that are ignored
#     )
    profileFaces = profileCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(35, 35),
#        maxSize=(50, 50)
    )

    # retângulo
    for (x, y, w, h) in frontalFaces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,255), 2)
        cv2.putText(frame, 'Frontal face', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 1)
    # for (x, y, w, h) in sideFaces:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (190,190,190), 2)
    #     cv2.putText(frame, 'Side face', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (190,190,190), 1)
    for (x, y, w, h) in profileFaces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (168,168,168), 2)
        cv2.putText(frame, 'Profile face', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (168,168,168), 1)


    cv2.imshow('''Detectando faces (frontal/side/profile) - pressione a tecla "Q" para sair''', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # key == q (quit)
        break

video_capture.release()
cv2.destroyAllWindows()


# https://towardsdatascience.com/a-guide-to-face-detection-in-python-3eab0f6b9fc1