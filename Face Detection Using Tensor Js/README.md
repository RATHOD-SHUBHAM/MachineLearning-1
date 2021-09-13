# Face Detection On Browser Using TensorFlow.js

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
