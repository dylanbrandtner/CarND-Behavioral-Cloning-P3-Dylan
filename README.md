# **Behavioral Cloning using a Convolutional Neural Network** 

## Goals: 
* Use the simulator to collect data of good driving behavior
* Build, a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./examples/Centerdriving.png "Center Driving"
[image3]: ./examples/Centerdriving_reversed.png "Center Driving Reversed"
[image4]: ./examples/LeftRecovery.png "Left Recovery"
[image5]: ./examples/RightRecovery.png "Right Recovery"
[image6]: ./examples/center_2018_08_21_15_37_30_274.jpg "Center Cam"
[image7]: ./examples/left_2018_08_21_15_37_30_274.jpg "Left Cam"
[image8]: ./examples/right_2018_08_21_15_37_30_274.jpg "Right Cam"
[image9]: ./examples/center_cropped.jpg "Cropped Cam"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used a convolutional neural network with a lambda layer to normalize the data, a cropping layer to reduce noise, 5 convolutional layers with RELU activation functions, a flattening layer, and 4 fully connected layers.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.  I used 20 epochs as the validation loss was no longer decreasing after that amount.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I captured two laps each of center lane driving in both directions, and a single lap of recovering from the left and right sides of the road. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with an existing model used in autonomous driving. I used the neural network used by Nvida as it was already a proven way to autonomously operate a vehicle.   

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set and ran my model in the simulator.  I found that my first model had a _very_ small validation loss (less than 0.01), but only preformed well on staight paths and without hitting the edges of the track.  It was unable to recover from any deviations from center driving.  Instead of making major adjustments to the architecture, I instead augmented the data.  This will be covered more in section 3 below as data collection and augmentation had the biggest impact on my results.     

I didn't have any major issues with overfitting, and since Nvidia's network did not contain any dropout, I decided not to introduce it myself. 

After augmenting the dataset with enough information, and setting the epochs to an amount where the validation loss leveled out, I reached a reasonable stopping point.  

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My final model architecture is based on the convolutional neural network used by the [autonomous driving team at Nvidia](https://devblogs.nvidia.com/deep-learning-self-driving-cars/):

Here is a visualization of the architecture:
![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the two laps of center lane driving in the opposite direction:

![alt text][image3]

I keep each lap in a separate directory so that I could exclude parts of my full data set if necessary.  Also, as the simulator took some getting used to, I didn't want to contaimate "good" data, with any failed runs.

I preprocessed this data by normalizing it (dividing it by 255) and then mean centering it (subtract 0.5), and randomly shuffled the data set and put 20% of the data into a validation set.  

Here is where my first testing began, but the results were not particularly good.  I noticed that the model seemed to only preform well when the vehicle was in the center of the road (which makes sense since that is the only training data it had).  Anytime it encoutered an edge or rolled slightly over it, the car would veer sharply off the track. 

To augment the data set with a bit more variatey, I added in the left and right angles camera images, and applied a correction of 0.2 to the steering angle to account for location of the camera.  Here is what the 3 different angles look like from the perspective of the car:

Center:
![alt text][image6]
Left:
![alt text][image7]
Right:
![alt text][image8]

I also noticed that in places with a lot of extra scenery, the performance was the worst.  Thus, I also cropped off the top 70 pixels and bottom 25 pixels of the images, as these were simply more noise for the model to be distracted by.  Below you can see the center camera image with the cropped area highlighed in red:  
![alt text][image9]

At this point, the vehicle could somtimes make it around the track, but again, if any correction steered the vehicle into a barrier too far, it could not recover.  It encoutered this scenario on some of the sharper turns.  

To combat this, I recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to do so if it ever encountered an edge.  These images show what a recovery looks like from both sides:

![alt text][image4]
![alt text][image5]

This improved my results, and to some degree, I could also force the car into a bad siutation by taking temporary manual control and driving it off onto the curb.  Unless I drove _over_ the curb, the model was able to recover.   

After this collection process, I had 22,257 total samples (include the 3 camera anlges). 

I used this training data for training the model. The ideal number of epochs seemd to be 20 as evidenced by a levling off in validation loss after this point.  I used an adam optimizer so that manually training the learning rate wasn't necessary.  My final validation loss was 0.0221.

## Result

### Car Driving a Full Autonomous Lap
Here is a video of the car driving fully autonomously around the track:
<iframe width="560" height="315" src="https://www.youtube.com/embed/NFF8ITAfV18" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

### Car Recovering from Manual Intervention
In this second video, I manually intervened a couple times to force the car onto the curb.  In all cases, it recovered:
