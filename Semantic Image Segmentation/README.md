# Semantic Image Segmentation.

Most of the time, I just experimented with object detection and recognition, which creates a bounding box around specific detected objects in an image. 
However, I later came across another technique that can give a precise outline of an object that has been detected in an image. 
This technique is known as Image Segmentation.

Followed this great [article](https://towardsdatascience.com/semantic-image-segmentation-with-deeplabv3-pytorch-989319a9a4fb) by Vinayak Nayak to get a better understanding of how segmentation work using pytorch and deeplab v3.

Steps:
1. we use cv2 to set up a video session and prepare output screens
2. we load deeplabv3 model and predict labels. (person class is with label 15)
3. We apply segmentation (labels) map and gaussian blur. 
4. we also apply custom background using the same technique.

