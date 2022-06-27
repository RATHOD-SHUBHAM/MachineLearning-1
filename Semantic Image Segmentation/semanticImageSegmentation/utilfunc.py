import cv2
import torch
import torchvision


class utilfunc:
    '''
        https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/pytorch_vision_deeplabv3_resnet101.ipynb#scrollTo=together-default
        https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        # model variants
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)

        to(device): https://stackoverflow.com/questions/63061779/pytorch-when-do-i-need-to-use-todevice-on-a-model-or-tensor
    '''

    # todo: Using the below code we can download the model from torch-hub and use it for our segmentation task.
    @staticmethod
    def load_model():
        # train on the GPU or on the CPU, if a GPU is not available
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        model.to(device).eval()
        return model


    '''
        OpenCv reads image as BGR format but while rendering we need to show format in RGB
        https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
    '''

    @staticmethod
    def grab_frame(cap):
        if not cap.isOpened():
            print("cannot open camera")
            exit()

        # Given a video capture object, read frames from the same and convert it into RGB
        print("cap: ", cap.read())
        _, frame = cap.read()
        print("frames__: ", frame)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    '''
        https://pytorch.org/vision/stable/transforms.html#scriptable-transforms
        
        pytorch normalize image : https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        https://www.geeksforgeeks.org/how-to-normalize-images-in-pytorch/
        
        Transform: https://pytorch.org/vision/stable/transforms.html
        Compose(transforms): Composes several transforms together.
        ToTensor() : Convert a PIL Image or numpy.ndarray to tensor.
        Normalize(mean, std[, inplace]): Normalize a tensor image with mean and standard deviation.
        
        unsqueeze: Returns a new tensor with a dimension of size one inserted at the specified position.
    '''

    '''
        # sample execution (requires torchvision)
        # https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/pytorch_vision_deeplabv3_resnet101.ipynb#scrollTo=elementary-acrylic
        
        from PIL import Image
        from torchvision import transforms
        input_image = Image.open(filename)
        input_image = input_image.convert("RGB")
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')
        
        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0)
    
    '''

    # todo: Prediction
    @staticmethod
    def get_pred(img, model):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # define standard transform that needs to be done at interface time
        imagenet_stats = [[0.485, 0.456, 0.406], [0.485, 0.456, 0.406]]
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=imagenet_stats[0],
                                             std=imagenet_stats[1])
        ])


        # All pre-trained models expect input images in mini-batches of 3-channel RGB images of shape (N, 3, H, W),
        # where N is the number of images, H and W are height and width of the two images respectively.

        # Since our video capture object captures single frames, it’s output is (3, H, W).
        # We therefore unsqueeze the tensor along the first dimension to make it (1, 3, H, W).

        input_tensor = preprocess(img).unsqueeze(0)
        print("unsqueeze: ",input_tensor)
        input_tensor = input_tensor.to(device)
        print("input_tensor: ",input_tensor)

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_tensor = input_tensor.to('cuda')
            model.to('cuda')

        # Make the predictions for labels across the image
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
            print("output: ", output)

            # do a argmax along the 1st dimension to obtain the labels map
            # which is of the same height and width as the original image with a single channel.
            # This mask could be used for segmentation.
            output_predictions = output.argmax(0)
            print("output+pred: ", output_predictions)

        # return the prediction
        return output_predictions.cpu().numpy()


