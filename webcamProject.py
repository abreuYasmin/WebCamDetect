import cv2 as cv

# aprendendo a usar a webcam
webcam = cv.VideoCapture(0)

while True:
    cam, frame = webcam.read()

    cv.imshow('Imagem WebCamera', frame)

    if cv.waitKey(1) == ord('s'):
        break
 
webcam.release()
cv.destroyAllWindows()

# nao ta fechando a janela
