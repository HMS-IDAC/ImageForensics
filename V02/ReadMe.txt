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

test_nn.py
    For a representative of each class in the test set, finds nearest neighbors among another
    representative of the same class and one representative for each of the other classes.

test_pairwise.py
    For a representative of each class in the test set, measures similarity w.r.t. another
    representative of the same class and one representative for each of the other classes.

Models.py
    Class containing siamese network model.

embedding.py
    Script to visualize embedding on TensorBoard.

image_distortions.py
    Functions that compute image distortions during training.

SemanticSegmentation/generate_features_for_segmentation.py
    Uses a trained siamese net model to generate features for semantic segmentation.

SemanticSegmentation/train_rf_from_features.py
    Trains a random forest using the features from the siamese network.

SemanticSegmentation/deploy_rf_from_features.py
    Uses the trained random forest, and features computed from the siamese net model, to segment images.

ToolBox/imtools.py
    Utility image analysis/processing tools.

ToolBox/ftools.py
    Utility file I/O and handling tools.



Developed by:
Marcelo Cicconet
marceloc.net
