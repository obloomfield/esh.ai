import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from model.discrim import Discriminator
from model.generator import Generator

def text_converter(text):
    idx_map = {
        "solid": 33,
        "floral": 55,
        "spotted": 31,
        "graphics": 25,
        "plaid": 51,
        "striped": 27,
        "red": 6,
        "yellow": 2,
        "green": 42,
        "cyan": 53,
        "blue": 29,
        "purple": 23,
        "brown": 4,
        "white": 40,
        "gray": 57,
        "black": 14,
        "many colors": 49,
        "necktie": 0,
        "scarf": 38,
        "exposure": 8,
        "collar": 10,
        "scarf": 38,
        "placket": 47
    }
    to_change = []
    text = set(text)
    for word in text:
        if word in idx_map:
            idx_map[word] += 1
        elif word == "shirt": 
            to_change.append(16)
        elif word == "sweater":
            to_change.append(17)
        elif word == "t-shirt":
            to_change.append(18)
        elif word == "outerwear":
            to_change.append(19)
        elif word == "suit":
            to_change.append(20)
        elif word == "tank top":
            to_change.append(21)
        elif word == "dress":
            to_change.append(22)
        elif word == "no sleeves":
            to_change.append(35)
        elif word == "short sleeves":
            to_change.append(36)
        elif word == "long sleeves":
            to_change.append(37)
        elif word == "v-shape":
            to_change.append(44)
        elif word == "round":
            to_change.append(45)
        elif owrd == "other shapes":
            to_change.append(46)

    one_hot = [0]*(59)
    for key in idx_map:
        i = idx_map[key]
        one_hot[i] = 1
    for i in to_change:
        one_hot[i] = 1

    return one_hot

def visualize(gen, disc, text, artsy_index):
    #gen, text => showcasing an example set of image gens.
    #gen_output = Tensor(256, 265, 3)
    #disc_output = float
    z = tf.random.normal([1,256], stddev=(1.0*artsy_index))
    one_hot = text_converter(text)
    img = gen(one_hot, z)
    img = np.asarray(img)
    img = np.squeeze(img, axis = 0)
    img = tf.keras.preprocessing.image.array_to_img(img)
    im = Image.open(img)
    im.show()

def load_weights(model, pth):
    # load weights from path
    model.load_weights(pth)

def main():
    input_text = ["red", "floral"]
    g = Generator()
    print('LOADING GENERATOR')
    load_weights(g, 'weights/generator.pth')
    d = Discriminator()
    print('LOADING DISCRIMINATOR')
    load_weights(d, 'weights/discriminator.pth')
    print('GENERATING IMAGE')
    visualize(g, d, input_text, 1)
