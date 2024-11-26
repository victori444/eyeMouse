Inspiration:
The concept was born to break barriers in digital accessibility. We wanted to create a solution for people with limited dexterity or struggle with using keyboards and mice.

What it does:
EyeMouse uses a combination of face tracking and hand gesture recognition to simulate mouse movements and keyboard interactions.
- Face tracking: Navigate your cursor with precise facial movements using AI-powered facial landmark detection.
     - Cursor: Nose tracking
     - Clicking: Eye wink
     - Scrolling: Tilting your head: Left -> up, Right -> Down
     - Dragging: Raising your eyebrows
- Hand gesture recognition: Controls the keyboard and responds to fingertip proximity to type.
- Voice-to-text feature: Enable it by opening your mouth and it will write down what you say.
- Sound effects: Added sound effects for actions.

How we built it:
We integrated:
- OpenCV
     - capture video from the webcam
     - process frames to detect features & cursor position
     - draw graphics (virtual keyboard) on video feed
- MediaPipe
     - framework developed by Google to build pipelines to process audio/video/sensor data
     - implement hand tracking & face mesh detection
- Speech recognition library
     - speech to text conversion
     - voice-based control to interact with system
- Python libraries
     - PyAutoGUI
          - Python library to automate mouse & keyboard input
          - simulate key presses & cursor movements based on the detected facial/hand movements
     - Pygame
          - sound playback associated to actions
     - Numpy
          - calculate distances between landmarks on facial/hand mesh (eg. Eye Aspect Ratio)
            
Challenges we ran into:
Developing a solution that balances precision and responsiveness was challenging.
- Synchronizing face and hand gestures without interference required a lot of testing.
- Making sure that our winking feature doesn't recognize blinking.
- Ensuring the virtual keyboard was intuitive and responsive as we encountered many problems with the space button.
- Achieving real-time performance while maintaining low latency.

Accomplishments that we're proud of:
We are proud of creating a program that works pretty accurately with facial movements, hand and voice recognition.

What we learned:
We learned many new features with python like face and body tracking using machine learning combined with multimedia.

What's next for EyeMouse:
- Enhance AI Models: Improve gesture detection accuracy and include more customizable gestures.
- Implement more features like shortcuts.
- Integration with other devices.
- Making it more accessible as an extension or an app.
