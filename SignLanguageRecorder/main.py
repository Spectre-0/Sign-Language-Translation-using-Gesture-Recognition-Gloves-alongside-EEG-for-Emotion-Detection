from utils.ArduinoInterpreter import ArduinoInterpreter
from utils.DisplaySignLanguage import DisplaySignLanguage
from utils.SaveFile import saveData
import asyncio, signal, json, sys
from multiprocessing import Process, Manager, Event
import tkinter as tk
import numpy as np
import pickle
import string
import os, re
import time

# ----------------------------------------------------------------------------------------------------------

# ---------------------------------- Settings ----------------------------------

# ----------------- Interpreter Settings ------------------

# Read Data From Glove [Left, Right]
hand_setup = np.array([True, True])

# ------------------- Recorder Settings -------------------

# words = ["hi"]
words = list(string.ascii_lowercase)

# Time to get ready
get_ready_timer = 1
# Time for action
action_timer = 3


# ----------------------------------------------------------------------------------------------------------

# -------------------------------- Exit Function -------------------------------

def cleanup_and_exit(processes):
    """Handles cleanup on Ctrl+C."""
    print("\nCaught Ctrl+C! Terminating processes...")

    time.sleep(2)

    # Terminate child processes
    for p in processes:
        p.terminate()
        p.join()

    sys.exit(0)

# ----------------------------------------------------------------------------------------------------------

# -------------------- Initialize Interpreter and Read Data --------------------

def connectToArduino(isConnected, handData):
    # Create a new event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        arduinoInstance = ArduinoInterpreter(isConnected, handData)
        loop.run_until_complete(arduinoInstance.run(hand_setup))
    except Exception as e:
        print(f"Arduino connection error: {e}")
    finally:
        loop.close()

# ----------------------------------------------------------------------------------------------------------

# ------------------------ Display Sign Language Window ------------------------

def displaySigns(isConnected, handData, signData, stop_event):
        
    def handleCompletion(data: dict):
        for word in data.keys():
            wordData = {"word": data[word]['name'], "valid": data[word]['valid'], "handData": []}
            timestamps = data[word]['timestamps']

            if timestamps[0] == timestamps[1]:
                continue

            for log in list(handData):
                if log[-1] > timestamps[0] and log[-1] < timestamps[1]:
                    wordData["handData"].append(log)

            signData.append(wordData)

        curr_path = os.getcwd()
        folder_path = os.path.join(curr_path, "data")
        saveData(signData, folder_path)

    while not all(isConnected):
        time.sleep(2)

    root = tk.Tk()
    root.title("Sign Language")
    root.geometry("854x480")
    app = DisplaySignLanguage(root, words=words, handleCompletion=handleCompletion, gready_timer=get_ready_timer, action_timer=action_timer)

    # Bind space key to toggle valid
    root.bind('<space>', app.toggle_valid)
    root.mainloop()

    if not app.hasCompleted:
        handleCompletion(app.get_data())

    # Signal the main process to stop all processes
    stop_event.set()

# ----------------------------------------------------------------------------------------------------------

# --------------------------------- Code At Init -------------------------------

# ------------------------ Monitor Glove Connections --------------------------

def monitor_connections(isConnected, stop_event):
    """Monitor glove connections with a clean, updating status display."""
    previous_states = [False, False]
    
    # Initial status line
    print("\nConnection Status:")
    print("Glove 1: Disconnected")
    print("Glove 2: Disconnected")
    
    while not stop_event.is_set():
        current_states = list(isConnected)
        
        if current_states != previous_states:
            # Move cursor up 3 lines (status header + 2 gloves)
            print("\033[3A", end='')
            
            # Reprint the header and current status
            print("\nConnection Status:                      ")
            for i in range(len(current_states)):
                status = "Connected   " if current_states[i] else "Disconnected"
                print(f"Glove {i+1}: {status}                 ")
                
        previous_states = current_states.copy()
        time.sleep(0.5)

if __name__ == "__main__":
    with Manager() as manager:
        isConnected = manager.list(hand_setup == False)  # Initialize as False until truly connected
        handData = manager.list([])
        signData = manager.list([])  # Changed from {} to [] to match expected type

        stop_event = Event()

        p1 = Process(target=connectToArduino, args=(isConnected, handData))
        p2 = Process(target=displaySigns, args=(isConnected, handData, signData, stop_event))
        p3 = Process(target=monitor_connections, args=(isConnected, stop_event))

        processes = [p1, p2, p3]

        # Register signal handler
        signal.signal(signal.SIGINT, lambda sig, frame: cleanup_and_exit(processes))

        # Start processes
        for p in processes:
            p.start()

        stop_event.wait()

        cleanup_and_exit(processes)