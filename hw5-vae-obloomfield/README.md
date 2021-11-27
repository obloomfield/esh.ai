# HW4: VARIATIONAL AUTOENCODER


##### Overview

This program trains on the MNIST number image dataset, and using a VAE Neural Network is able to generate a new handwritten digit (0-9) with low loss. Through this exercise, it is found that the VAE model performs somewhat better than the CVAE, and doesn't take much more time to train.

VAE final **LOSS**: 16941.935547
CVAE final **LOSS**: 17720.564453

[Tensorflow](https://www.tensorflow.org/) is used alongside [numpy](https://numpy.org/) to build the network.


##### Known Bugs

- Although the images are shown to be very accurate, the reported loss is often uncomfortably high, due to 


##### How to Run:

If you run ```python3 assignment.py``` or ```python3 assignment.py --is_cvae```  with all of the packages in ```requirements.txt``` pip install -ed on python 3.8+, it should show, after heavy calculation, show the end loss of each of the respective models and output sample generated images to /outputs.

##### How to modify:

- Modify the hidden_dim, latent_size hyperparameters for variable performance.