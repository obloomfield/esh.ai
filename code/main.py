import tensorflow as tf
import numpy as np
import random
from tensorflow.keras import Model
from preprocess import get_data
from model.discrim import Discriminator
from model.generator import Generator


def save_weights(model, pth):
    # save weights to specified path
    model.save_weights(pth)
    
def load_weights(model, pth):
    # load weights from path
    model.load_weights(pth)

def train(g, d, train_imgs, train_text, batch_sz, res, artsy_index):
    """
    Runs through one epoch - all training examples.

    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)

    i = 0
    for i in range(0, len(train_imgs), batch_sz):
        cur_imgs = train_imgs[i:i + batch_sz]
        cur_labels = train_text[i:i + batch_sz]
        # finding random label from batch, using generic gan setup tutorial
        rand_labels = train_text[np.random.randint(train_text.shape[0], size=(batch_sz)),:]
        z = tf.random.normal([batch_sz, res], stddev=(1.0*artsy_index))
        
        #GENERATOR GRADIENTS - 
        with tf.GradientTape() as tape:
            #generator

            fake_gen = g(cur_labels, z)
            
            score = d(fake_gen, cur_labels)
            g_loss = g.loss(score)
            
        gradients = tape.gradient(g_loss, g.trainable_variables)
        optimizer.apply_gradients(zip(gradients, g.trainable_variables))
            
            
        #DISCRIMINATOR GRADIENTS - 
        with tf.GradientTape() as tape:
           
            # ~~ generator:
            
            fake_gen = g(cur_labels, z)
            
            # ~~ discriminator on: 
            
            # fake trick : fake img, real labels
            fake_trick_score = d(fake_gen, cur_labels) # (tries to see if generator flat-out tricks the discrim)
            # all real : real img, real labels
            all_real_score = d(cur_imgs, cur_labels) 
            # rand label  : real img, randomly chosen labels
            rand_label_score = d(cur_imgs, rand_labels)
            
            # loss based on all the scores
            d_loss = d.loss(fake_trick_score, all_real_score, rand_label_score)
            
        gradients = tape.gradient(d_loss, d.trainable_variables)
        optimizer.apply_gradients(zip(gradients, d.trainable_variables))

        print(f'Batch: {i} | Gen Loss: {g_loss} | Disc Loss: {d_loss}')

def test( g, d, test_imgs, test_text):
    g_loss_acc = 0
    d_loss_acc = 0
    for (cur_img, cur_label) in list(zip(test_imgs, test_text)):
        
        # ~~ generator:
        fake_gen = g(cur_label, cur_img)
        
        # fake trick : fake img, real labels
        fake_trick_score = d(fake_gen, cur_label) # (tries to see if generator flat-out tricks the discrim)
        # all real : real img, real labels
        all_real_score = d(cur_img, cur_label) 
        # rand label  : real img, randomly chosen labels
        rand_label_score = d(cur_img, random.choice(test_text))
        
        g_l = g.loss(fake_trick_score)
        g_loss_acc += g_l
        
        d_l = d.loss(fake_trick_score,all_real_score,rand_label_score)
        d_loss_acc += d_l
    return g_loss_acc, d_loss_acc
    

def main():
    print('STARTING PREPROCESSING')
    img_train, img_test, lbl_train, lbl_test = get_data()
    print('PREPROCESSING COMPLETE')
    
    g = Generator()
    d = Discriminator()
    
    # load_weights(g,'weights/generator.pth')
    # load_weights(d,'weights/discriminator.pth')
    
    NUM_EPOCHS = 50
    BATCH_SIZE = 30
    RESOLUTION = 256
    ARTSY_INDEX = 0.9

    print('TRAINING MODEL')
    for i in range(NUM_EPOCHS):
        print("epoch: ", i)
        train(g, d, img_train, lbl_train, BATCH_SIZE, RESOLUTION, ARTSY_INDEX)
    print('TRAINING COMPLETE')
    
    gen_acc, disc_acc = test(g, d, img_test, lbl_test)
    print(f'Gen: {gen_acc} | Disc: {disc_acc}')
    
    save_weights(g,'weights/generator.pth')
    save_weights(d,'weights/discriminator.pth')
    
    

if __name__ == "__main__":
    main()