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

    def optimize(tape: tf.GradientTape, model: tf.keras.Model, loss: tf.Tensor) -> None:
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

    for i in range(batch_sz, len(train_imgs), batch_sz):
        cur_imgs = train_imgs[i-batch_sz:i]
        cur_labels = train_text[i-batch_sz:i]
        # finding random label from batch, using generic gan setup tutorial
        rand_labels = train_text[np.random.randint(train_text.shape[0], size=(batch_sz)),:]
        z = tf.random.normal([batch_sz, res], stddev=(1.0*artsy_index))
        
        with tf.GradientTape(persistent=True) as tape:
            # generated images
            fake_gen = g(cur_labels, z)
            
            # pass real images with real text through discriminator
            logits_real = d(cur_imgs, cur_labels)
            # pass real images with random text through discriminator
            logits_rand = d(cur_imgs, rand_labels)
            # pass fake images with real text through discriminator
            logits_fake = d(fake_gen, cur_labels)

            g_loss = g.loss(logits_fake)
            d_loss = d.loss(logits_fake, logits_real, logits_rand)
        optimize(tape, g, g_loss)
        optimize(tape, d, d_loss)

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
    BATCH_SIZE = 100
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