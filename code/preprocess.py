import glob
from PIL import Image
import tensorflow as tf 
import scipy.io
import numpy as np

def get_data(): 
    images = glob.glob("ClothingAttributeDataset/images/*")
    labels = glob.glob("ClothingAttributeDataset/labels/*")
    label_res = []
    image_res = []
    for i in range(len(images)):
        img = Image.open(images[i])
        img = img.resize((256, 256))
        rgb_tensor = tf.keras.preprocessing.image.img_to_array(img)
        image_res.append(rgb_tensor)
        temp = []
        for label in labels:
            mat = scipy.io.loadmat(label)
            mat = mat["GT"]
            if "sleeve" in label or "neckline" in label:
                arr = [0]*3
            elif "category" in label:
                arr = [0]*7
            else:
                arr = [0, 0]
            specific = mat[i][0]
            if not(np.isnan(specific)):
                arr[int(specific) - 1] = 1
            temp += arr
        label_res.append(temp)

    return image_res, label_res
