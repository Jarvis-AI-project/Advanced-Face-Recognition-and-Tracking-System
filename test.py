import cv2
import os
# Capture video from webcam and make dataset
class MakeDataset():
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.count = 0
        self.person_name = input("Enter person name: ")
        self.num_images = int(input("Enter number of images: "))
        self.path = "data/" + self.person_name
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def make(self):
        for _ in range(self.num_images):
            _, frame = self.video.read()
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                self.count += 1
                cv2.imwrite(self.path + "/" + str(self.count) + ".jpg", frame)
                print("Image saved")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def __final__(self):
        self.video.release()
        cv2.destroyAllWindows()


MakeDataset().make()
