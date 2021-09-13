# Face Detection On Browser Using TensorFlow.js

Check out the result [here](https://youtu.be/O62iUMlP_Jc).

With TensorFlow.js we can develop ML models in JavaScript, and use ML directly in the browser or in Node.js.

## How it Works
  1. Run existing models.
  2. Retrain existing models.
  3. Develop ML with JS.

## Model Used
 * Simple Face Detection: Detects faces in. images using a 'Single Shot Detector' Architecture with a custom encoder (BlazeFace).

## Blazeface detector
  * Blazeface is a lightweight model that detects faces in images. 
  * Blazeface makes use of a modified Single Shot Detector architecture with a custom encoder. 
  * The model may serve as a first step for face-related computer vision applications, such as facial keypoint recognition.

### Standalone Script tag:

    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/blazeface"></script>

## MediaDevices.getUserMedia()
Gives access to web cam & video feed.

The MediaDevices.getUserMedia() method prompts the user for permission to use a media input which produces a MediaStream with tracks containing the requested types of media.

That stream can include, for example, a video track , an audio track , and possibly other track types.

It returns a Promise that resolves to a MediaStream object. If the user denies permission, or matching media is not available, then the promise is rejected with NotAllowedError or NotFoundError respectively.


## Step Performed:
  1. Get access to web cam & video feed using getUserMedia() api.
  2. Detect Face using TensorFlow.js.
  3. Make Predection using blazeface.
  4. Dispaly Result.


## Run:
    Install live Server or nodemon.
Open index.html on web browser.
