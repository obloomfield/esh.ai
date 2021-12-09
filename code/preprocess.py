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
        img_lab = []
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
            img_lab += arr
        label_res.append(img_lab)
    
    #returns image train and test split as well as the corresponding label train and test split
    #return image_train, image_test, label_train, label_test
    return image_res[:1501], image_res[1501:], label_res[:1501], label_res[1501:]
