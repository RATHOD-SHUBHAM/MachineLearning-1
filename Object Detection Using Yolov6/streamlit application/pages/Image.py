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
# File uploader
def save_uploadedfile(uploadedfile):
    with open(os.path.join("/Users/shubhamrathod/PycharmProjects/ObjDet_yolov6/YOLOv6/data/images/", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to /Users/shubhamrathod/PycharmProjects/ObjDet_yolov6/YOLOv6/data/images/".format(uploadedfile.name))

image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
col1, col2, col3 = st.columns([1, 1, 1])
if image_file is not None:
    file_details = {"FileName":image_file.name,"FileType":image_file.type}
    with col1:
        st.write("File Details")
        st.write(file_details)
    save_uploadedfile(image_file)
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
        "source": "/Users/shubhamrathod/PycharmProjects/ObjDet_yolov6/YOLOv6/data/images/" + image_file.name,
        "yaml": "/Users/shubhamrathod/PycharmProjects/ObjDet_yolov6/YOLOv6/data/coco.yaml",
        "img_size": 640,
        "conf_thres": st.sidebar.slider('Configuration Threshold', 0.0, 1.0, 0.25),
        "iou_thres": st.sidebar.slider('IOU Threshold', 0.0, 1.0, 0.45),
        "max_det": st.sidebar.slider('Maximal detections per image', 500, 5000, 1000),
        "device": 0,  # device to run our model i.e. 0 or 0,1,2,3 or cpu
        "save_txt": st.sidebar.selectbox('Do you want to save the Image',
     ('True', 'False')),
        "save_img": True,
        "classes": None,  # filter detection by classes
        "agnostic_nms": False,  # class-agnostic NMS
        "project": os.path.join("/Users/shubhamrathod/PycharmProjects/ObjDet_yolov6/YOLOv6/runs/inference"),
        "name": 'exp',
        "hide_labels": False,
        "hide_conf": False,
        "half": False,
    }

    # ------------------------------------------------------------------------------------------------------------------


    # todo:  Inference on Single Image
    if not os.path.isdir("/Users/shubhamrathod/PycharmProjects/ObjDet_yolov6/YOLOv6/runs/inference"):
        os.makedirs("/Users/shubhamrathod/PycharmProjects/ObjDet_yolov6/YOLOv6/runs/inference")

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


    # read image
    try:
        img_src = cv2.imread(args['source'])
        # If img_src is None, the function immediately terminates. Otherwise, it continues until the end.
        assert img_src is not None, f'Invalid Image'
    except Exception as e:
        print("Invalid path.")

    start = time.time()  # calculating the infernce time
    image, img_src = Inferer.precess_image(img_src, inferer.img_size, inferer.model.stride, args['half'])
    detection = inferer.infer(image, img_src)
    # print("detection: ",detection)
    end = time.time()

    total_time = end - start

    fps_txt = "{:.2f}".format(1 / end)  # display total time as frame per second
    print("Frame per second: ", fps_txt)
    # st.write("Frame per second: ", fps_txt)

    with col2:
        st.write("Labels")
        for *xyxy, conf, cls in reversed(detection):
            # print("conf: ",conf)
            # print("cls: ",cls)
            class_num = int(cls)  # integer class

            label = None if args['hide_labels'] else (
                inferer.class_names[class_num] if args['hide_conf'] else f'{inferer.class_names[class_num]} {conf:.2f}')

            print("label: ", label)

            st.write(label)

            Inferer.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label,
                                       color=Inferer.generate_colors(class_num, True), fps=fps_txt)

    image = np.asarray(img_src)

    print("Inferencing time: ", total_time)
    with col3:
        st.write("Inference Time")
        st.write(total_time)

    if args['save_img'] == "True":
        image_name = args['source'].split('/')[-1]
        cv2.imwrite(args['project'] + image_name, image)

    # cv2.imshow("Output Image", image)
    #input image
    col1, col2 = st.columns([1, 1])
    col1.image(args['source'])
    # output image
    col2.image(image, channels="RGB")
    # cv2.waitKey(0)
