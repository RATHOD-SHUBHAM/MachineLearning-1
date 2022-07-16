# todo: Import Libraries
import os
import sys
from imageInference import Inferer
import cv2
import time
import numpy as np
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

image_file = st.file_uploader("Upload An Video",type=['mp4'])
# col1, col2, col3 = st.columns([1, 1, 1])
if image_file is not None:
    file_details = {"FileName":image_file.name,"FileType":image_file.type}
    # with col1:
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




    # todo: Inference on Video

    video_path = "/Users/shubhamrathod/PycharmProjects/ObjDet_yolov6/test.mp4"

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

    # https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
    video = cv2.VideoCapture(video_path)

    # Capture frame-by-frame
    ret, img_src = video.read()

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    output = cv2.VideoWriter('output.mp4', fourcc, 30, (img_src.shape[1], img_src.shape[0]))

    while True:

        if ret:

            start = time.time()  # calculating the infernce time
            image, img_src = Inferer.precess_image(img_src, inferer.img_size, inferer.model.stride, args['half'])
            detection = inferer.infer(image, img_src)
            end = time.time()

            total_time = end - start
            print("Inferenceing Time: ", total_time)

            fps_txt = "{:.2f}".format(1 / end)  # display total time as frame per second
            print("Frame per second: ", fps_txt)

            for *xyxy, conf, cls in reversed(detection):
                class_num = int(cls)  # integer class

                label = None if args['hide_labels'] else (
                    inferer.class_names[class_num] if args[
                        'hide_conf'] else f'{inferer.class_names[class_num]} {conf:.2f}')

                print("label: ", label)

                Inferer.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label,
                                           color=Inferer.generate_colors(class_num, True), fps=fps_txt)

            image = np.asarray(img_src)

            output.write(image)
            ret, img_src = video.read()

        else:
            break

    output.release()
    video.release()

    # display output
    cap = cv2.VideoCapture('/Users/shubhamrathod/PycharmProjects/ObjDet_yolov6/output.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

