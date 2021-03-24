'''
ECE276A WI20 HW1
Stop Sign Detector
A53295675 Yunhsiu Wu 
'''

import os, cv2
import numpy as np
from skimage.measure import label, regionprops
from gaussian_model import Gaussian_model


class StopSignDetector():
    def __init__(self):
        '''
			Initilize your stop sign detector with the attributes you need,
			e.g., parameters of your classifier
		'''
        #single gaussian model parameters are obtained in gaussian_model.py
        self.COLOR_STOP_SIGN_RED = {}
        self.COLOR_STOP_SIGN_RED['mean']        = np.array([ 22.20063474,  14.1578286 , 151.38942607])
        self.COLOR_STOP_SIGN_RED['mean_other']  = np.array([125.56838205, 118.35595876, 107.90070133])
        self.COLOR_STOP_SIGN_RED['cov']         = np.array([[ 458.7189013 ,  377.20925136,  287.19151111],
                                                           [ 377.20925136,  389.6662935 ,  297.58856068],
                                                           [ 287.19151111,  297.58856068, 2595.36567865]])
        self.COLOR_STOP_SIGN_RED['cov_other']   = np.array([[4848.76955546, 3380.69340819, 2408.78334666],
                                                           [3380.69340819, 3088.2825218 , 2777.42226017],
                                                           [2408.78334666, 2777.42226017, 3163.31729258]])
        self.COLOR_STOP_SIGN_RED['prior']       = 0.010426400078926715
        self.COLOR_STOP_SIGN_RED['prior_other'] = 0.9895735999210733
        
        
    def segment_image(self, img):
        '''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
        img_size = np.shape(img)
        img = np.reshape(img, (-1,3))
        mask_img = np.zeros(np.shape(img)[0])
        
        #calculate probability of each color
        pixel_COLOR_STOP_SIGN_RED = Gaussian_model(img, self.COLOR_STOP_SIGN_RED['mean'], self.COLOR_STOP_SIGN_RED['cov'], self.COLOR_STOP_SIGN_RED['prior'])
        pixel_other = Gaussian_model(img, self.COLOR_STOP_SIGN_RED['mean_other'], self.COLOR_STOP_SIGN_RED['cov_other'], self.COLOR_STOP_SIGN_RED['prior_other'])
        
        #classify pixels into 1s and 0s
        for pixel in range(np.shape(img)[0]):
            if pixel_COLOR_STOP_SIGN_RED[pixel] > pixel_other[pixel]:
                mask_img[pixel] = 1
                
        mask_img = cv2.dilate(mask_img, (21,21), iterations = 5)
        mask_img = cv2.erode(mask_img, (21,21), iterations = 5)
        
        mask_img = mask_img.reshape(img_size[:2])
        
        return mask_img.astype('uint8')
    
    
    def get_bounding_box(self, img):
        '''
			Find the bounding box of the stop sign
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
        boxes = []
        img_mask = self.segment_image(img)
        img_label = label(img_mask)
        countours = regionprops(img_label)
        
        for countour in countours:
            box = countour.bbox
            x, y, w, h = box[1], box[0], box[3]-box[1], box[2]-box[0]
            area_ratio = countour.extent
            score = 0
            score = score + 10 * (1 - abs( w / h - 1) )
            if w > 30 and h > 30:
                score = score + 5
            if area_ratio < 0.75 and area_ratio > 0.5:
                score = score + 3
            elif area_ratio > 0.7:
                score = score + 2 * ( abs(area_ratio-0.7) / 0.7 )
            elif area_ratio > 0.2: 
                score = score + 2 * ( abs(area_ratio-0.2) / 0.3 )
            
            if score > 15.5 :
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 3)
                boxes.append( [x, img.shape[0]-y-h, x+w, img.shape[0]-y] )
        
        boxes.sort( key = lambda x :x[0])
       
        return boxes


if __name__ == '__main__':
    folder = "trainset"
    my_detector = StopSignDetector()
    
    # For whole data set
    '''
    for filename in os.listdir(folder):
		#detect image and show result
        img = cv2.imread(os.path.join(folder,filename))
        img_mask = my_detector.segment_image(img)
        img_resize = cv2.resize(img, (800, 600))
        img_mask_resize = cv2.resize(img_mask * 255, (800, 600))
        cv2.imshow('image', img_resize)
        cv2.waitKey(10)
        cv2.imshow('image mask', img_mask_resize)
        cv2.waitKey(10)
        cv2.destroyAllWindows()
        
        box = my_detector.get_bounding_box(img)
        #print(box)
        img_resize = cv2.resize(img, (800, 600))
        cv2.imshow('image', img_resize)
        cv2.waitKey(10)
        cv2.destroyAllWindows()
    '''
    
    # For certain data
  
	#detect image and show result
    filename = r'./trainset/186.jpg'
    img = cv2.imread(filename)                          
    img_mask = my_detector.segment_image(img)
    img_resize = cv2.resize(img, (800, 600))
    img_mask_resize = cv2.resize(img_mask * 255, (800, 600))
    cv2.imshow('image', img_resize)
    cv2.waitKey(10)
    cv2.imshow('image mask', img_mask_resize)
    cv2.waitKey(10)
    #cv2.destroyAllWindows()
    
    box = my_detector.get_bounding_box(img)
    print(box)
    img_resize = cv2.resize(img, (800, 600))
    cv2.imshow('image masked', img_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
		#Display results:
		#(1) Segmented images
		#	 mask_img = my_detector.segment_image(img)
		#(2) Stop sign bounding box
		#    boxes = my_detector.get_bounding_box(img)
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope