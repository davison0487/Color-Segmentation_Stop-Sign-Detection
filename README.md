# Color Segmentation – Stop Sign Detection

- 1.stop_sign_detector.py
Main detection file.
This file uses function written in "gaussian_model.py" to calculate probability. Parameters obtained in "gaussian_param.py" are pre-inputed in this file.

- 2.gaussian_model.py
Calculate probability with single gaussian model.

- 3.gaussian_param.py
Retrieving mutlivariate single guassian model parameters.

- 4.labeltool.py
Labeling training set.

# Results

## Segmentation and Detection
This section we will present some result of masked image and boundary boxes, along with some discussions.
Let’s take a look at an easy example. This image has a clear vision for the stop sign, we can easily detect stop sign just by the model.
 
![image](image/fig1) Detection result for 16.jpg 
Fig. 2 Masked image for 16.jpg
In 38.jpg (Fig.3, Fig.4), there are two stop signs with a few non-stop sign red objects. As we can see in the masked image, we have classified one stop sign and one cut into two-halves. Unfortunately, the cut-halved stop sign is being filtered out by scoring mechanism. However this result is acceptable, we can barely see red pixels at the side of the sign, only half of the stop sign should not pass the scoring system. Also, some small masks did not pass the test as expected.
 
Fig. 3 Detection result for 38.jpg
 
Fig. 4 Masked image for 38.jpg
Now we try an image with more red objects and examine the function of scoring mechanism when the classifying model did not work out very well. The masked image by the model showed the red car and fire hydrant. Thanks to filtering system, we have successfully detected correct result without the interference of car and the hydrant.
 
Fig. 5 Detection result for 41.jpg
 
Fig. 6 Masked image for 41.jpg

Fig.7 and Fig.8 is a failure example for low quality image, 43.jpg is a relatively small image with the size of 432x230. Although we obtain a stop sign-like shape in masked image, there is a gap between upper and bottom half. This gap is disastrous for scoring mechanism, two halves obviously will not pass the filter. I believe better image quality with decent image process would definitely help improving detection results.
 
Fig. 7 Detection result for 43.jpg
 
Fig. 8 Masked image for 43.jpg

Finally, let’s try image with stop sign similar objects. Although there are no stop signs in this image, the result give us two stop signs. Observing mask areas, they indeed follow the property we used in scoring mechanism even if they are far from octagon. We have this results because we are using only the basic properties of octagon such as height, width and area ratio.
 
Fig. 9 Detection result for 186.jpg
 
Fig. 10 Masked image for 186.jpg

# Conclusion
In conclusion, we implemented single multivariate Gaussian model for color segmentation and basic octagon properties for detection. We have fine results for good condition images and acceptable result for mediocre images. Detection for low quality Images is what we should try to resolve in future works. Obtaining great quality test data will substantially boost results, however in real life, we have to take noises and various camera condition into account. Preprocessing image and implementing rigorous properties of octagon are two suggestive way to make progress.
