# Captcha-Breaker

## Goal
To train a neural network to correctly identify which characters are being displayed in the CAPTCHA.

## Background
CAPTCHA is an acronym for Completely Automate Public Turing test to tell Computers and Humans Apart. CAPTCHAs are primarily used by websites trying to stop bots from accessing web pages and potentially scraping data.

###### Sample CAPTCHA Image
<p align="center">
  <img src="https://github.com/slieb74/Captcha-Breaker/blob/master/images/captcha.png">
</p>

## Dataset
We used the Claptcha to randomly generate CAPTCHAs with added noise and distortion using 62 characters (0-9, Aa-Zz). To train our model to recognize each character, we generated 1,000 random CAPTCHAs for each of our 62 characters. Every 5th CAPTCHA would be funneled into a test dataset, while every 8th was funneled into a validation dataset for later use. The rest of the CAPTCHAs were added to the dataset we used to train our model. This resulted in the following dataset split: training data (70%), test data (20%), and validation data (10%).

We repeated this process when creating multi-character CAPTCHAs, generating 20,000 samples each for 4, 5, and 6 character CAPTCHAs.

## Preprocessing
Our single character CAPTCHAs were initially shaped 28x28x3, but since we only were focusing on grayscale images, we flattened each CAPTCHA into a 28x28 image. Then, since we were feeding the entire training set into our neural network, we reshaped the images into 784x1 shape, so when we were feeding our training data into the model, the shape was 43400x784. To normalize our data, we divided each image by 255, so that the resulting numbers were always between 0 and 1.

## Neural Network
Our approach was to first train a multilayer perceptron neural network to classify each character. Then, we would implement YOLOv3 (You Only Look Once), a real-time object detection library developed by Joseph Redmon and Ali Farhadi (https://pjreddie.com/media/files/papers/YOLOv3.pdf). This would allow us to detect where each character was in the multi-character CAPTCHAs, and then correctly predict them using the original neural network we trained.

###### Tensorboard Diagram of our Neural Network
<p align="center">
  <img src="https://github.com/slieb74/Captcha-Breaker/blob/master/images/tensorboard.png">
</p>

The first step of this worked exceedingly well. We created a neural network with an input layer containing 784 nodes (shape of the image), 3 hidden dense layers, and an output layer using 'softmax' activation because it is a categorical classification problem. For the other layers, we found that using 'relu' activation resulted in far greater accuracy than 'tanh'. The 'adam' optimizer proved to be the best for our model, consistently providing better results than the 'SGD' (Stochastic Gradient Descent) optimizer. We played around adding dropout layers after some of all of our dense layers, but found that it resulted in too great a decrease in accuracy to include in our final model. After numerous iterations training our model with different batch sizes and the number of epochs, we settled on 50 epochs and a batch size of 100.

###### Training and Validation Loss and Accuracy
<p align="center">
  <img src="https://github.com/slieb74/Captcha-Breaker/blob/master/images/Screen%20Shot%202018-10-12%20at%203.32.08%20PM.png.png">
</p>


Our neural network correctly identified single character CAPTCHAs in our test set with 91.1% accuracy. Precision was 92%, while Recall and our F1 Score were each 91%, so our model was very good at limiting both false negatives and false positives.

###### Confusion Matrix (due to visibility constaints on 62x62 confusion matrix, we broke it down into 3 sections: numbers, lowercase letters, uppercase letters)
<p align="center">
  <img src="https://github.com/slieb74/Captcha-Breaker/blob/master/images/cm_nums.png" height="450" width="450">
</p>
<p align="center">
  <img src="https://github.com/slieb74/Captcha-Breaker/blob/master/images/cm_lower.png" height="450" width="450">
</p>  
<p align="center">
  <img src="https://github.com/slieb74/Captcha-Breaker/blob/master/images/cm_upper.png" height="450" width="450">
</p>
We then trained the YOLOv3 model to detect where each character was in multi-character CAPTCHAs. However, after training this model for 16 hours, it did not work as planned and we had to scrap it due to time constraints. We needed to find a different method to detect objects because going through the image pixel by pixel would not give a clear picture of what character was being looked at and would be extraordinarily time consuming. 

Our solution was to instead create a moving window algorithm, which will break the image into K equal slices, and then for each slice try to predict which character was in the image. We could then average all of the predictions to see which character was there. This is a highly inaccurate method because many of the slices either contained parts of two characters, or just a sliver of the one character shown. In our research, we found that the upper bound for accuracy for an algorithm like this was under 1%, yet we still gave it a try to replace the YOLOv3 method. Our accuracy ended up being middling as well, as we were unable to crack the 1% threshold. 

## Next Steps
* Get YOLOv3 working so we can detect where each character is located within the image, and then classify it correctly
* Train our model to solve audio CAPTCHAs in addition to text-based
