**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image0]: ./output_images/example_car.png
[image1]: ./output_images/example_not_car.png
[image2]: ./output_images/car_example_hog.jpg
[image3]: ./output_images/not_car_example_hog.jpg

[image4]: ./output_images/sample_pipeline_output_testimages_1.jpg
[image5]: ./output_images/sample_pipeline_output_testimages_2.jpg

[video1]: ./project_video_marked.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in `get_hog_features()` and `extract_features_image()` of the file called `features.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![test][image0] 
![test][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of 
`orient=9` 
`pix_per_cell=8`
`cell_per_block=2`

Here is vizualization for car and not car image:
![hog car][image2]
![hog not car][image3]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and color spaces (most parameters were default form the lesson material)  fo all the color spaces `YCrCb` gave most validation i.e. test accuracy in classfier training.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in cell 3 of notebook `VehicleDetection.ipynb`).  
I trained a linear SVM using HOG features for each channel of `YCrCb` color space image. Also used the histogram features.
Here is the summary of training:
`32.5 Seconds to train SVC...
Test Accuracy of SVC =  0.9817
Train Accuracy of SVC =  1.0`

### Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

All the code for sliding window is in `pipeline.py`
`Large window over bottom half of image`
` w = slide_window(img, x_start_stop=[None, None],y_start_stop=[np.int(image_length*0.5), image_length],xy_window=(256, 256), xy_overlap=(0.75, 0.75))`

`Medium window over bottom half of image`
`w = slide_window(img, x_start_stop=[None, None], y_start_stop=[np.int(image_length*0.5), np.int(image_length*0.85)], xy_window=(128, 128), xy_overlap=(0.75, 0.75))`

`Small window over bottom half of image`
w = slide_window(img, x_start_stop=[None, None], y_start_stop=[np.int(image_length*0.5), np.int(image_length*0.70)], xy_window=(64, 64), xy_overlap=(0.75, 0.75))`

![pipeline][image4]
![pipeline][image5]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

See pipeline images above. It shows windows for vehicles based on pipeline before (classifier output) and after filtering for heat maps.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [video1](./project_video_marked.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

I am also tracking detectiosn in cars.py and for next frame I include the cars detected in this frame as well as previous frames.  Cars are not displayed (retired) when then do not show up in threshold frames (FRAME_COUNT_THRESHOLD = 6)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One thing that worked for me was to use probability classifier and set a threshold for match to > 90% probabilty.  This only worked because the data used to train the classifier was of high quality.  This along with the fact that I setup overlap of 90% between sliding windows and setting threshold of 3 eliminated pretty much all false positives.  Another problem was missed detections. To overcome this tracking cars and displaying them even when are not detected with some tolerance before its retired.
