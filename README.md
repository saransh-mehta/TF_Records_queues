# image


You are building TWO models using Tensorflow to recognize apple fruit in images. 

1. You will NOT use pretrained models. You will do the training yourself. Create a training data set of atleast 100 images. Also create a validation data set of 10 images. Make sure the training dataset has images of different sizes and apples of different colors.
2. Convert the training data image to a tfrecord.
3. Create your OWN activation function (using tensorflow lambdas) instead of the normal ones. Justify why you are choosing this - if you choose others and this worked better, tell us about all the ones that didnt work !
4. Create two CNN models - one using your own activation function and the other using Relu. Create the models by training them on the training dataset.
5. Compare which one was better using the validation dataset.
