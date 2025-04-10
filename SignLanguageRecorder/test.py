import asyncio
import time
import os
import sys

# Import your ArduinoInterpreter
from utils.ArduinoInterpreter import ArduinoInterpreter

async def main():
    """Minimal test script to connect to gloves"""
    # Initialize basic connection lists
    isConnected = [False, False]  # [left, right]
    handData = []
    
    # Set up which hands to use - trying both by default
    hands = [True, True]
    
    print("BSL Glove Connection Test")
    print("=========================")
    print("Attempting to connect to gloves...")
    
    # Create Arduino interpreter instance
    arduino_interpreter = ArduinoInterpreter(isConnected, handData)
    
    # Create a separate task for the Arduino connection
    arduino_task = asyncio.create_task(arduino_interpreter.run(hands))
    
    # Monitor connection status
    try:
        while True:
            # Clear the terminal
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Print connection status
            print("BSL Glove Connection Test")
            print("=========================")
            print(f"Left Glove: {'Connected' if isConnected[0] else 'Disconnected'}")
            print(f"Right Glove: {'Connected' if isConnected[1] else 'Disconnected'}")
            
            # Print raw data if available
            if handData:
                print("\nLast data point received:")
                print(handData[-1])
                print(f"Total data points received: {len(handData)}")
            else:
                print("\nNo data received yet.")
            
            # Check the addresses being used
            if hasattr(arduino_interpreter, 'LEFT_IMU_ADDRESS'):
                print(f"\nLeft glove address: {arduino_interpreter.LEFT_IMU_ADDRESS}")
                print(f"Right glove address: {arduino_interpreter.RIGHT_IMU_ADDRESS}")
            
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
        # Cancel the task
        arduino_task.cancel()
        try:
            await arduino_task
        except asyncio.CancelledError:
            pass
        
        print("Connection test complete.")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())