Welcome to the Sign Language Recording App!

-- Description --
This app is aimed to record sign language data from the custom-made gloves.

The app performs 2 main functions:
- Establishes a Connection with the gloves and reads data from them
- Displays words/letters to be signed and records the action in a given time window

-- How It Works --
When you run the app, after a connection has been established with the glove/s, a windows should pop up, displaying a start button.

On the top left:
	- Valid label is visible, indicating the current state of the recording
	- Toggleable using the spacebar
	- It is intended to be used whenever an action has some **knocking** present

On the top right:
	- A timer indicating either:
	- The time left to get prepared
	- The time left to sign

In the middle:
	- Large text, representing the word or letter to be signed

Button:
	- Allows for start/stop of current sign
	- Whenever the sign is stopped, both timers will restart, hence allowing for a restart in recording of the sign

Initially, when the start button is pressed, it will initialize the sign language recording.
The app will iterate through each word inside the provided list one after the other.
You may exit the app at any time by pressing the close button (i.e. the X in the top right), the recorded signs will be saved in a file under the "data" folder.

-- Python Packages --
- Bleak (version 0.22.3)
- NumPy (version 1.26.4)

Disclaimer: Bleak package doesn't install when specifying version (i.e. pip install bleak==0.22.3), hence resort to standard installation process.

-- How To Run --
After successful installation of above python packages, you can run the app from you command prompt, by calling:
"python main.py"

-- Settings --
Within the main.py file, there is a limited amount of settings, allowing for you to specify:
- Which glove to connect to (Left/Right/Both)
- Words to be signed (Has to be a list of strings)
- Size of Preparation & Action Time Windows

-- Future Work --
Probably add an option to start from a given word in the list instead of always at the beginning.

-- More --
For more information and/or support please contact Wiktor, or cry trying to figure what, where and when.
I did try to make it readable, but life do be a bish ¯\_(ツ)_/¯.

Also "dev" folder just has a jupyter notebook I tested/worked on, so ignore it.
