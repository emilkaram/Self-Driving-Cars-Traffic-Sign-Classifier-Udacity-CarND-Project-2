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

 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


