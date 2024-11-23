"use strict";

let faceMesh;
let video;
let faces = [];
let options = { maxFaces: 1, refineLandmarks: false, flipHorizontal: false };

// Custom cursor element
let cursorElement;

function preload() {
  // Load the FaceMesh model
  faceMesh = ml5.faceMesh(options);
}

function setup() {
  createCanvas(640, 480);
  
  // Create the webcam video and hide it
  video = createCapture(VIDEO);
  video.size(640, 480); // Match the video size with the canvas size
  video.hide();
  
  // Hide the system cursor
  noCursor();

  // Create the custom cursor element (the green dot)
  cursorElement = createDiv('');
  cursorElement.id('customCursor'); // Apply the id from CSS
}

function draw() {
  // Draw the webcam video
  image(video, 0, 0, width, height);

  // Continuously detect faces in real-time
  faceMesh.detect(video, gotFaces);

  // If there are faces detected, track the nose tip and move the custom cursor
  if (faces.length > 0) {
    // Get the first detected face (you can handle multiple faces if needed)
    let face = faces[0];

    // Nose tip is usually the 4th point in the keypoints array (index 4)
    let nose = face.keypoints[4];
    
    // Map the nose position to the canvas size (taking video size into account)
    let noseX = map(nose.x, 0, video.width, 0, width);
    let noseY = map(nose.y, 0, video.height, 0, height);

    // Move the custom cursor element to the nose position
    cursorElement.position(noseX - cursorElement.width / 2, noseY - cursorElement.height / 2);

    // Optionally, visualize the nose point with a small green circle
    fill(0, 255, 0);  // Green color
    noStroke();
    ellipse(noseX, noseY, 10, 10);  // Draw a circle on the nose position
  }
}

// Callback function for when faceMesh detects faces
function gotFaces(results) {
  // Save the output of detected faces
  faces = results;
}

