train.py
    Script to train/test the model on synthetic data.
    Expects training data to be grayscale images, of size 256x256.

test.py
    Script to test the model on real data.
    Expects test folder to contain subfolders with images that are manipulations of the same image.
    For example, if test folder is Test, and its subfolders are Test_i, all images in 
    Test_i should be manipulations of the same base image I_i, and base images I_i and I_j
    should be different images when i is different from j.
    All images on Test should be grayscale and of size 128x128.

Models.py
    Class containing siamese network model.

embedding.py
    Script to visualize embedding on TensorBoard.

image_distortions.py
    Functions that compute image distortions during training.


Developed by:
Marcelo Cicconet
marceloc.net