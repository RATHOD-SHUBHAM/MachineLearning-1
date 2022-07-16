# todo: Import Libraries
import os
import sys
from imageInference import Inferer
import cv2
import time
import numpy as np
from IPython.display import display, Javascript
from base64 import b64decode, b64encode
import PIL
import io
import streamlit as st

# Once the requirements are installed fetch the pretrained model
# todo: Fetching pretrained model
# create a folder called weights to store the weights file
if not os.path.isdir("/Users/shubhamrathod/PycharmProjects/ObjDet_yolov6/YOLOv6/weights"):
    os.makedirs("/Users/shubhamrathod/PycharmProjects/ObjDet_yolov6/YOLOv6/weights")

# print(sys.path)
sys.path.append('/Users/shubhamrathod/PycharmProjects/ObjDet_yolov6/YOLOv6')


# ------------------------------------------------------------------------------------------------------------------


st.sidebar.title("Configuration")


# ------------------------------------------------------------------------------------------------------------------


# setting up arguments
# got the arguments from /Users/shubhamrathod/PycharmProjects/ObjDet_yolov6/YOLOv6/tools/infer.py
# """ Inference process, supporting inference on one image file or directory which containing images.
#    Args:
#        weights: The path of model.pt, e.g. yolov6s.pt
#        source: Source path, supporting image files or dirs containing images.
#        yaml: Data yaml file, .
#        img_size: Inference image-size, e.g. 640
#        conf_thres: Confidence threshold in inference, e.g. 0.25
#        iou_thres: NMS IOU threshold in inference, e.g. 0.45
#        max_det: Maximal detections per image, e.g. 1000
#        device: Cuda device, e.e. 0, or 0,1,2,3 or cpu
#        save_txt: Save results to *.txt
#        save_img: Save visualized inference results
#        classes: Filter by class: --class 0, or --class 0 2 3
#        agnostic_nms: Class-agnostic NMS
#        project: Save results to project/name
#        name: Save results to project/name, e.g. 'exp'
#        line_thickness: Bounding box thickness (pixels), e.g. 3
#        hide_labels: Hide labels, e.g. False
#        hide_conf: Hide confidences
#        half: Use FP16 half-precision inference, e.g. False
# """

args = {
    "weights": "/Users/shubhamrathod/PycharmProjects/ObjDet_yolov6/YOLOv6/weights/yolov6n.pt",
    "source": "/Users/shubhamrathod/PycharmProjects/ObjDet_yolov6/YOLOv6/data/images/image1.jpg",
    "yaml": "/Users/shubhamrathod/PycharmProjects/ObjDet_yolov6/YOLOv6/data/coco.yaml",
    "img_size": 640,
    "conf_thres": st.sidebar.slider('Configuration Threshold', 0.0, 1.0, 0.25),
    "iou_thres": st.sidebar.slider('IOU Threshold', 0.0, 1.0, 0.45),
    "max_det": st.sidebar.slider('Maximal detections per image', 500, 5000, 1000),
    "device": 0,  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "save_txt": False,
    "save_img": True,
    "classes": None,  # filter detection by classes
    "agnostic_nms": False,  # class-agnostic NMS
    "project": os.path.join("/Users/shubhamrathod/PycharmProjects/ObjDet_yolov6/YOLOv6/runs/inference"),
    "name": 'exp',
    "hide_labels": False,
    "hide_conf": False,
    "half": False,
}

# todo: Inference on Webcam
## Webcam Helper Functions
# webcame helper func: https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial/blob/master/yolov4_webcam.ipynb

# function to convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream
def bbox_to_bytes(bbox_array):
    """
    Params:
            bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
    Returns:
          bytes: Base64 image byte string
    """
    # convert array into PIL image
    bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
    iobuf = io.BytesIO()
    # format bbox into png for return
    bbox_PIL.save(iobuf, format='png')
    # format return string
    bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))

    return bbox_bytes


#  WebCam Inference

inferer = Inferer(
    source=args['source'],
    weights=args['weights'],
    device=args['device'],
    yaml=args['yaml'],
    img_size=args['img_size'],
    half=args['half'],
    conf_thres=args['conf_thres'],
    iou_thres=args['iou_thres'],
    classes=args['classes'],
    agnostic_nms=args['agnostic_nms'],
    max_det=args['max_det']
)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# label for video
label_html = 'Capturing...'
# initialze bounding box to empty
bbox = ''
count = 0
while True:
    ret, img_src = cap.read()
    start = time.time()  # calculating the infernce time
    image, img_src = Inferer.precess_image(img_src, inferer.img_size, inferer.model.stride, args['half'])
    detection = inferer.infer(image, img_src)
    end = time.time()

    total_time = end - start

    fps_txt = "{:.2f}".format(1 / end)  # display total time as frame per second
    print("Frame per second: ", fps_txt)

    for *xyxy, conf, cls in reversed(detection):
        class_num = int(cls)  # integer class

        label = None if args['hide_labels'] else (
            inferer.class_names[class_num] if args['hide_conf'] else f'{inferer.class_names[class_num]} {conf:.2f}')

        print("label: ", label)

        Inferer.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label,
                                   color=Inferer.generate_colors(class_num, True), fps=fps_txt)

    image = np.asarray(img_src)
    print("Inferencing time: ", total_time)

    cv2.imshow('frame', image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
