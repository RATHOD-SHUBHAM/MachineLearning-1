#todo: import libraries
import cv2
import matplotlib.pyplot as plt
#import the util function
from utilfunc import *
import numpy as np


# todo: load the deeplabv3 model
model = utilfunc.load_model()


#todo: background image substitution
BLUR = False
bg_path = "/Users/shubhamrathod/PycharmProjects/semanticImageSegmentation/bg.jpeg"
# Read the background image to memory
bg_image = cv2.imread(bg_path)
bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)


'''
Using opencv to interface a webcam for reading input from screen
opencv videoCapture object is used to get image input for video
'''
# todo: Start a video cam session
video_session = cv2.VideoCapture(0)




# todo: Input pre processing step
"""
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html

figure: has figsize: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib.pyplot.figure
figsize = change the figure size

Ticks are the values used to show specific points on the coordinate axis. It can be a number or a string.
https://www.geeksforgeeks.org/python-matplotlib-pyplot-ticks/

axes: If there is more than one subplot
axes: https://matplotlib.org/stable/api/axes_api.html#matplotlib.axes.Axes

"""

# Define two axes for showing the mask adn the actual video on real time
'''
So basically we set up two subplots one to see the blurred version and another to look at the labels mask which the deeplab model will predicted. 
We switch on the interactive mode for pyplot and display the image captured directly at the very beginning of the stream.
'''
fig , (ax1 , ax2) = plt.subplots(1, 2, figsize = (15 , 8))

# Axis labels, title, and legend
ax2.set_title("Image Mask")

# Ticks and tick labels
# set the ticks to none for both the axes
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])


# create 2 image object for the picture for the axes defined above
img_1 = ax1.imshow(utilfunc.grab_frame(video_session))
print("img_1: ",img_1)
img_2 = ax2.imshow(utilfunc.grab_frame(video_session))
print("img_2: ",img_2)


# switch on the interactive mode in matplotlib
plt.ion() # Enable interactive mode.
plt.show()

# todo: Human Segmentation

# Define the kernel size for applying Gaussian Blur
blur_value = (51, 51)

# Read frames from the video, make realtime predictions and display the same
while True:
    frame = utilfunc.grab_frame(video_session)

    # Ensure there's something in the image (not completely blank)
    if np.any(frame):
        print("frames: ",frame)
        # Read the frame's width, height, channels and get the labels' predictions from utilities
        width , height , channels = frame.shape
        labels = utilfunc.get_pred(frame , model)
        print("labels: ",labels)

        # The labels extracted out of the image are two dimensional.
        # We need to replicate this mask across all three channels i.e. RGB.
        # So, we use numpy to repeat this segmentation mask across all three channels.
        if BLUR:
            # Wherever there's empty space/no person, the label is zero
            # Hence identify such areas and create a mask (replicate it across RGB channels)
            mask = labels == 0
            # https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
            mask = np.repeat(mask[: , : , np.newaxis] , channels , axis = 2)

            # https://appdividend.com/2020/09/19/python-cv2-filtering-image-using-gaussianblur-method/
            # Apply the Gaussian blur for background with the kernel size specified in constants above
            blur = cv2.GaussianBlur(frame , blur_value,  0)
            print("blur: ", blur)
            frame[mask] = blur[mask]
            ax1.set_title("Blurred Image")
        else:
            # The PASCAL VOC dataset has 20 categories of which Person is the 16th category
            # Hence wherever person is predicted, the label returned will be 15
            # Subsequently repeat the mask across RGB channels
            mask = labels == 15
            mask = np.repeat(mask[ : , : , np.newaxis] , 3 , axis = 2)

            # Resize the image as per the frame capture size
            bg = cv2.resize(bg_image, (height, width))
            bg[mask] = frame[mask]
            frame = bg
            ax1.set_title("Background Changed Video")

        # set the data of the two image to frame and mask values
        img_1.set_data(frame)
        img_2.set_data(mask * 255)
        plt.pause(0.02)

    else:
        break

# todo: Empty the cache and switch off the interactive mode
torch.cuda.empty_cache()
plt.ioff()



