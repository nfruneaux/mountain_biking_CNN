# mountain_biking_CNN

![visualized_image_5](https://github.com/nfruneaux/mountain_biking_CNN/assets/72471698/dc2311e5-9802-47af-9687-6430f4ac40c3)

(A) Background: 

This is my first NN model created from own data, with own code. Originally initiated in fall 2023 after studying Karpathy zero-to-hero series, Lecun NYU deep learning course, and exploring famous examples like hand-digit recognition using MNIST data set. I was very excited to learn more about neural net models, and thought it would be good to build a project around something I love: mountain biking. With the idea that this could be a first step / aspect in developing more advanced applications, with an interest on autonomous agents and robots.

(B) The MTB NN Project Summary: 

Image segmentation CNN trained on custom data to detect labels including: Trail, Sky, Trees 

This project contains the following main components
  (i) ~20 training images, which are 1st person POV screenshots from mountain biking videos from Pacific Northwest trails
  (ii) XML labels for each training image, representing labelled boxes drawn on each training image identifying the following:
        (a) Trees
        (b) Trail
        (c) Sky
        (d) Biker
        (e) Bike

  (iii) A set of validation images, meaning other screenshots from mountain biking videos, that have not been used to train the model
  (iv) Python code which configures a convolution neural network to identify the labelled features, trained in batches

In addition, there is a folder named "Trained Model Output" which contains a trained CNN model (weights, optimizer, etc.) which can be imported into the visalization part of the code (instead of training a new model from scratch). This way one can re-use the current best trained model, and/or test it on new validation images. This model was only trained on the three main categories of focus in this project: Trees, Trail, Sky.

Inside the python code, in the directory summary section at the top, there is also a line "save_dir" which will create a folder and save a trained model, if training is initialized and completed in full. Presumably this folder would only be created in your local clone folder.

Regarding the training process, a number of optimizations were tried including different CNN parameters, different loss functions, etc. The current approach remains simple in the use of CNN parameters, and relies as well on some small hyperparamters in the loss function related to distribution loss and regularization. It is believed that these improve the model training but needs to be tested more thoroughly --- generally the conclusion has been that the overall architecture integrity and data qualiity is more meaninful than hyperparameters. 
