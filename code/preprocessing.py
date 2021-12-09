import tensorflow as tf
# from tensorflow import keras
import matplotlib.pyplot as plt
# import pathlib
import tensorflow as tf


dataset_url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
data_dir = tf.keras.utils.get_file(
    origin=dataset_url, fname="flower_photos", untar=True
)


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, image_size=(180, 180), batch_size=64
)

print(type(dataset))
for data, labels in dataset.take(1):
    print(data.shape)
    print(labels.shape)

class_names = dataset.class_names
for images, labels in dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        a = plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")