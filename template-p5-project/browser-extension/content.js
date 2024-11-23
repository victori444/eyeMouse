"use strict";

// Webcam video element
let video;
let faces = [];
let faceMesh;
let options = { maxFaces: 1, refineLandmarks: false, flipHorizontal: false };

// Custom cursor element
let cursorElement;

function setup() {
  // Create the video element dynamically
  video = document.createElement('video');
  video.width = 320;  // Set desired width
  video.height = 240; // Set desired height
  video.autoplay = true; // Ensure the video plays automatically
  video.style.position = 'fixed'; // Fix the video in place (no scrolling)
  video.style.bottom = '10px'; // Position from bottom
  video.style.right = '10px';  // Position from right
  video.style.border = '2px solid white'; // Optional border for visibility
  video.style.zIndex = '9999'; // Ensure the video is on top of other content

  // Append the video element to the body
  document.body.appendChild(video);

  // Create the custom cursor element (e.g., a red dot)
  cursorElement = document.createElement('div');
  cursorElement.id = 'customCursor';
  cursorElement.style.width = '10px';
  cursorElement.style.height = '10px';
  cursorElement.style.backgroundColor = 'red'; // Red dot for the cursor
  cursorElement.style.borderRadius = '50%'; // Make it circular
  cursorElement.style.position = 'absolute'; // Use absolute positioning
  cursorElement.style.pointerEvents = 'none'; // Ensure it doesn't block interaction
  document.body.appendChild(cursorElement); // Append to the body

  // Start the FaceMesh model
  faceMesh = ml5.faceMesh(options, modelReady);

  // Hide the default system cursor
  document.body.style.cursor = 'none';

  // Request webcam access using getUserMedia
  navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
      video.srcObject = stream; // Set the video source to the webcam stream
    })
    .catch((err) => {
      console.error("Error accessing webcam: ", err);
    });
}

function modelReady() {
  console.log("FaceMesh model is ready.");
}

function draw() {
  // Continuously detect faces in real-time
  faceMesh.detect(video, gotFaces);

  // If there are faces detected, track the nose tip and move the custom cursor
  if (faces.length > 0) {
    // Get the first detected face
    let face = faces[0];

    // Nose tip is usually the 4th point in the keypoints array (index 4)
    let nose = face.annotations.nose_tip;

    // Map the nose position to the window size (taking video size into account)
    let noseX = map(nose[0][0], 0, video.width, 0, window.innerWidth);
    let noseY = map(nose[0][1], 0, video.height, 0, window.innerHeight);

    // Move the custom cursor element to the mapped nose position
    cursorElement.style.left = `${noseX - cursorElement.offsetWidth / 2}px`;
    cursorElement.style.top = `${noseY - cursorElement.offsetHeight / 2}px`;
  }
}

// Callback function for when faceMesh detects faces
function gotFaces(results) {
  // Save the output of detected faces
  faces = results;
}

// Helper function to map values (similar to p5.js)
function map(value, start1, stop1, start2, stop2) {
  return start2 + (stop2 - start2) * (value - start1) / (stop1 - start1);
}

// Initialize setup
setup();
