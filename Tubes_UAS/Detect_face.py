# from tkinter.font import names
import cv2

video = cv2.VideoCapture(0)
a = 0

recognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetect = cv2.CascadeClassifier(
    './xml_asset/haarcascade_frontalface_default.xml')
recognizer.read('training/training.xml') 
id = 0
# format text
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)

detect = int(input('masukkan id user '))
names = ['Anda Siapa?', 'akbar', 'iqbal', 'ranti', 'hisoka']
asist = ['????', 'ASD, PBO', 'ASD, PBO', 'ASD', 'tidak ada']

while True:
    check, frame = video.read()
    print(check)
    print(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        a = a+1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 100:
            if id == detect:
                name = names[id]
                asisten = asist[id]
            else:
                name = "Anda Siapa?"
                asisten = asist[0]
        else:
            name = "Anda Siapa?"
            asisten = asist[0]
            # conf = " {0}%".format(round(150-conf))
        cv2.putText(frame, str(name), (x+5, y-5),
                    fontFace, fontScale, fontColor, 2)
        # cv2.putText(frame, str(id), (x+10, y+h-10),
        #             fontFace, fontScale, fontColor, 2)
        cv2.putText(frame, str(asisten), (x+10, y+h-10),
                    fontFace, fontScale, fontColor, 2)
    cv2.imshow("wajah", frame)
    if (cv2.waitKey(1) == ord('q')):
        break
    print(a)
video.release()
cv2.destroyAllWindows()
