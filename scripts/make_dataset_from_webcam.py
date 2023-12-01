import cv2
import os
class MakeDataset():
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.count = 0
        self.person_name = input("Enter person name: ")
        # self.num_images = int(input("Enter number of images: "))
        self.path = 'Y:\\DATASETS\\face-recognition-tensorflow-test-data\\' + self.person_name
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def make(self):
        while True:
            _, frame = self.video.read()
            cv2.imshow("frame", frame)
            # if cv2.waitKey(1) & 0xFF == ord('s'):
            self.count += 1
            cv2.imwrite(self.path + "/" + self.person_name +'_' + str(self.count) + ".jpg", frame)
            print(f'Image saved: {self.count}', end='\r')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if self.count == 500:
                break

    def __final__(self):
        self.video.release()
        cv2.destroyAllWindows()

MakeDataset().make()
