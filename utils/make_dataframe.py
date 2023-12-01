from keras.preprocessing.image import ImageDataGenerator
from utils.list_images_inside_folder import list_images
from utils.extract_faces import extract_frontal_face
import cv2
import numpy as np


# function to make the image size uniform and make train and test object with labels
class dataframe():
    def __init__(self):
        self.deva_faces = deva_faces
        self.swar_faces = swar_faces
        self.neg_faces = neg_faces
        self.images, self.labels = list(), list()

        self.total_images = len(deva_faces)+len(swar_faces)+len(neg_faces)
        self.labels_list = ['Unknown', 'Devasheesh', 'Swarnim']

        print('Total images: ', self.total_images)

        self.datagen = ImageDataGenerator(
            # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=40,
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.2,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.2,
            shear_range=0.2,    # set range for random shear
            zoom_range=0.2,    # set range for random zoom
            horizontal_flip=True,   # randomly flip images
            fill_mode='nearest'  # set mode for filling points outside the input boundaries
        )

    def append_dataframe(self, images_list, label, augment=False, grayscale=False):
        count = 1
        for image_adr in images_list:
            try:
                # read the image
                image = cv2.imread(image_adr)
                # extract the face
                face, _ = extract_frontal_face(
                    image, grayscale=grayscale, size=(300, 300))

                print('Count: ', count, ' | Image address: ', image_adr, ' | Label: ',
                      self.labels_list[label], ' | Image shape: ', image.shape)
                if face is not None:
                    # print('Face type: ', type(face))
                    # print('Image shape: ', image.shape)
                    # print('Face shape: ', face.shape)
                    # print('Face: ', plt.imshow(face))

                    # append the face image to the list
                    if augment and range(10):
                        # using the image data generator to augment the images
                        for batch in self.datagen.flow(face.reshape(1, 300, 300, 3), batch_size=1):
                            # append the image to the list
                            self.images.append(batch[0])
                            self.labels.append(label)

                    else:
                        self.images.append(face)
                        self.labels.append(label)
                        count += 1
                else:
                    print(f'Face not found in {image_adr}')
                    continue
            except Exception as e:
                print(f'Error: {e} in {image_adr}')
                continue

    def make_dataframe(self):
        self.append_dataframe(self.neg_faces, 0, augment=False)
        self.append_dataframe(self.deva_faces, 1, augment=True)
        self.append_dataframe(self.swar_faces, 2, augment=True)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        print('Images shape: ', self.images.shape)
        print('Labels shape: ', self.labels.shape)
        return self.images/255.0, self.labels, self.labels_list


x = dataframe()
images, labels, labels_list = x.make_dataframe()

if __name__ == '__main__':
    deva_faces = list_images('./data/Devasheesh/')
    swar_faces = list_images('./data/Swarnim/')
    neg_faces = list_images('./data/negative-faces/')
    # neg_faces = list_images(r'Y:\DATASETS\negative-faces\VGG-Face2\data\test')
