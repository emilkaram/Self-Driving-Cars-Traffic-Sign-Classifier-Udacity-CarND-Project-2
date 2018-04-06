# **Traffic Sign Recognition** 

## Writeup

### This project is to build a traffic signal classifier using convolution neural network.
### The dataset from German Traffic sign (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### The test images used to test the model:

![](https://github.com/emilkaram/Udacity-CarND-Traffic-Sign-Classifier-Project2/blob/master/images/prediction.png)



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 0. Introduction:
This project is to build a traffic signal classifier using convolution neural network.
The dataset from German Traffic sign. 

You're reading it! and here is a link to my [project code](https://github.com/emilkaram/Udacity-CarND-Traffic-Sign-Classifier-Project2/blob/master/Traffic_Sign_Classifier%20-emil8.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.
I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set = 34799
* The size of the validation set = 4410
* The size of test set is = 12630
* The shape of a traffic sign image = (32, 32, 3)
* The number of unique classes/labels in the data set = 43

#### 2. An exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.
Classes/lables table and bar chart showing how the data distribution for each unique class in the trian , validation and test datasets.

##Class label
![Class label](https://github.com/emilkaram/Udacity-CarND-Traffic-Sign-Classifier-Project2/blob/master/images/classes.png)

##Class Disribution
![Class Distribution](https://github.com/emilkaram/Udacity-CarND-Traffic-Sign-Classifier-Project2/blob/master/images/class%20dist.png)

### Design and Test a Model Architecture

#### 1. Preprocessed the image data.

As a first step, I decided to normiliaze the images to range 0 to 1 to have consistent range and easy to model.
Then shuflle the trainging dataset
 


#### 2. Final model architecture is a modified version of the LeNet model 
Here is a diagram and describition of the final model.

 ![final model](https://github.com/emilkaram/Udacity-CarND-Traffic-Sign-Classifier-Project2/blob/master/images/Modified_LeNet.png)
 
###Implement the modfied LeNet neural network architecture.
Input The modified LeNet architecture accepts a 32x32x3 image as input, where 3 is the number of color channels.
Architecture Layer 1: Convolutional. The output shape = 28x28x18.
Activation. activation function =RELU.
Pooling. The output shape = 14x14x18.
Layer 2: Convolutional. The output shape = 10x10x48.
Activation. activation function =RELU.
Pooling. The output shape = 5x5x48.
Flatten. Flatten the output shape =5x5x48 = 1200 of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.
Layer 3: Fully Connected. This should = 120 outputs.
Activation. activation function =RELU.
Layer 4: Fully Connected = 84 outputs.
Activation. activation function =RELU.
Layer 5: Fully Connected (Logits)= 43 outputs (Classes).
Output Return the result of the 2nd fully connected layer.

#### 3. Describtion of how I trained my model:
Type of optimizer used: AdamOptimizer
The batch size =50 
Number of epochs =30
Learning rate =0.001
Activation functions = RELU
additional layers = maxpooling
accuracy operation = reduce_mean

 

#### 4. Describtion of the approach taken for finding a solution and getting the validation set accuracy > 0.93.
I 

My final model results were:
* training set accuracy of =1.000
* validation set accuracy =0.960
* test set accuracy of =0.951

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? original LeNet.
* What were some problems with the initial architecture? Does not have enough layers to tarin the model (underfitting)
* How was the architecture adjusted and why was it adjusted? 
 Adjustments made to LeNet model by adding layers Maxpooling.
* Which parameters were tuned? How were they adjusted and why? The batch size =50  , Number of epochs =30 parmater adjustment based on the preliminary results of the error and accuracy.
* What are some of the important design choices and why were they chosen? addong addition layer to convolution layer1 works well with this model.

 
 

### Test a Model on New Images

#### 1. Five new traffic signs found on the web and tested.

Here are five new traffic signs that I found on the web:

![](https://github.com/emilkaram/Udacity-CarND-Traffic-Sign-Classifier-Project2/blob/master/images/test_samples.png)


The first image might be difficult to classify because ...

#### 2. The model's predictions on new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![](https://github.com/emilkaram/Udacity-CarND-Traffic-Sign-Classifier-Project2/blob/master/images/prediction.png)


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 0.833. 


#### 3. Looking at the softmax probabilities for each prediction. T
he top 5 softmax probabilities for each image along with the sign type of each probability.  

![](https://github.com/emilkaram/Udacity-CarND-Traffic-Sign-Classifier-Project2/blob/master/images/prob.png)


Confusion Matrix
![](https://github.com/emilkaram/Udacity-CarND-Traffic-Sign-Classifier-Project2/blob/master/images/CM.png)


![](https://github.com/emilkaram/Udacity-CarND-Traffic-Sign-Classifier-Project2/blob/master/images/prediction.png)
 
The first image was "Right-of-way at the next intersection" and correctly prediction iwth 100% probabilty
The second image was "Keep right" and correctly prediction iwth 100% probabilty
The third image was "Turn left ahead" and was wongly predictied on the top and 2nd guess and got correct on 3rd guess with prob of 9%
The forth image was "General caution" and correctly prediction iwth 100% probabilty
the fith image was "Speed limit (60km/h)" and correctly prediction iwth 98% probabilty
The fith image was "No entry"and correctly prediction iwth 100% probabilty

####Consultion
The model works as expected with a good accuracy level


 
 
 
 


