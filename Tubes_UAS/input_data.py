import cv2

video = cv2.VideoCapture(0)  # untuk aktifin kamera realtime
a = 0

faceDetect = cv2.CascadeClassifier(
    './xml_asset/haarcascade_frontalface_default.xml')  # file untuk detect Haarcascade
id = input('masukkan id user ')  # menginputkan nama id gambar

while True:
    a = a + 1
    check, frame = video.read()  # membaca video camera real
    print(check)  # buat object frame untuk menangkap gambar dari kamera
    print(frame)
    # mengubah gambar jadi grayscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(grey, 1.3, 5)  # membaca Face Detect
    for (x, y, w, h) in faces:
        cv2.imwrite("DataSet/User."+str(id)+"."+str(a)+".jpg",
                    grey[y:y+h, x:x+w])  # penamaan data file
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Setting Rectangle Detection

    cv2.imshow("wajah", frame)  # Nampilin .. dengan nama Wajah

    # if key == ord('q'): #gambar berhenti jika kita klik q
    #     break
    # print(a)

    if (a > 100):  # gambar cuman disimpan kurang dari 100
        break
    print(a)

video.release()  # untuk menutup kamera
cv2.destroyAllWindows()  # menutup semua windows
