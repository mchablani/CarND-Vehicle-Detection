import cv2
import matplotlib.image as mpimg
import numpy as np
from skimage.feature import hog
from skimage import color, exposure

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features

class Features:
        
    # Define a function to compute color histogram features 
    # NEED TO CHANGE bins_range if reading .png files with mpimg!
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    # Define a function to compute binned color features  
    def bin_spatial(self, img, size=(64, 64)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel() 
        # Return the feature vector
        return features
    
    
    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self, imgs, color_space='YCrCb', spatial_size=(32, 32),
                         spatial_feat = False, hist_bins=32, orient=9, 
                         pix_per_cell=8, cell_per_block=2, hog_channel='ALL'):
        # Create a list to append feature vectors to
        features = []

        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = mpimg.imread(file)
            max_val = np.max(image)
            # print("max_val " + str(max_val))
            # scaled_img = (image/max_val)*255
            scaled_img = (image/max_val)
            
            features.append(self.extract_features_image(
                scaled_img, color_space, spatial_size,
                spatial_feat, hist_bins, orient, 
                pix_per_cell, cell_per_block, hog_channel))
        # Return list of feature vectors
        return features        
        
    
    
    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features_image(self, image, color_space='YCrCb', spatial_size=(32, 32),
                         spatial_feat = False, hist_bins=32, orient=9, 
                         pix_per_cell=8, cell_per_block=2, hog_channel='ALL'):
        file_features = []
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = self.bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        # Apply color_hist()
        hist_features = self.color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                                     orient, pix_per_cell, cell_per_block, 
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        file_features.append(hog_features)
        return np.concatenate(file_features)        
        
