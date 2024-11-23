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
  // Set the canvas size to the window size
  createCanvas(windowWidth, windowHeight); // Use full window size for the canvas
  
  // Create the webcam video with a fixed size
  video = createCapture(VIDEO);
  video.size(640, 480); // Fixed video size
  video.hide();
  
  // Hide the system cursor
  noCursor();

  // Create the custom cursor element (the green dot)
  cursorElement = createDiv('');
  cursorElement.id('customCursor'); // Apply the id from CSS
  cursorElement.style('width', '10px'); // Set a fixed size for the cursor
  cursorElement.style('height', '10px');
  cursorElement.style('background-color', 'red'); // Green dot for the cursor
  cursorElement.style('border-radius', '50%'); // Make it circular
  cursorElement.style('position', 'absolute'); // Use absolute positioning
  cursorElement.style('pointer-events', 'none'); // Ensure it doesn't block interaction
}

function draw() {
  // Clear the previous frame (clear canvas)
  background(0); // Optionally, you can use `clear()` if you prefer an empty canvas with no background

  // Draw the webcam video at a fixed size (top-left corner)
  image(video, 0, 0, 640, 480); // Fixed video size at (0, 0)

  // Continuously detect faces in real-time
  faceMesh.detect(video, gotFaces);

  // If there are faces detected, track the nose tip and move the custom cursor
  if (faces.length > 0) {
    // Get the first detected face
    let face = faces[0];

    // Nose tip is usually the 4th point in the keypoints array (index 4)
    let nose = face.keypoints[4];
    
    // Map the nose position to the window size (taking video size into account)
    let noseX = map(nose.x, 0, video.width, 0, windowWidth);
    let noseY = map(nose.y, 0, video.height, 0, windowHeight);

    // Move the custom cursor element to the mapped nose position
    cursorElement.position(noseX - cursorElement.width / 2, noseY - cursorElement.height / 2);


  }
}

// Callback function for when faceMesh detects faces
function gotFaces(results) {
  // Save the output of detected faces
  faces = results;
}

