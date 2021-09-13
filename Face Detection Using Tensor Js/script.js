// Store Video element
let video = document.getElementById("video");
let model;

// store canvas element
let canvas = document.getElementById("canvas");
// get the 2D image
let ctx = canvas.getContext("2d");


// getUserMedia API 
const setupCamera = () => {
    navigator.mediaDevices.getUserMedia({
        video: { width: 600, height: 400 } ,
        audio: false,
    }) // returns a stream object
    .then((stream) =>{
        video.srcObject = stream;
    });
};


const detectFaces = async () => {
    const prediction = await model.estimateFaces(video, false); 

    // console.log(prediction);


    ctx.drawImage(video, 0 , 0 , 600 , 400);

    // face Deteection
    // Draw blue Box
    prediction.forEach((pred) => {
        ctx.beginPath();
        ctx.lineWidth = "4";
        ctx.strokeStyle = "blue";

        // top left coordinates and height and width
        ctx.rect(
            pred.topLeft[0],
            pred.topLeft[1],
            pred.bottomRight[0] - pred.topLeft[0],
            pred.bottomRight[1] - pred.topLeft[1]
        );

        ctx.stroke();


        // add red land mark.
        ctx.fillStyle = "red";
        pred.landmarks.forEach((landmark) => {
            ctx.fillRect(landmark[0], landmark[1], 5 , 5 );
        }
        );

    });


}


setupCamera();

video.addEventListener("loadeddata", async () => {
    model =  await blazeface.load();
    // detectFaces(); // this work fine for image but while working with video call it 20/30 times per second
    
    // 1000/24 = 40 , so it will be called every 40 milli second
    setInterval(detectFaces,40);
})
