import os
import pickle
import re
import json

def find_next_filename(folder, pattern, extension):
    """
    Find the next available filename in the folder based on the pattern.
    :param folder: Path to the folder.
    :param pattern: Filename pattern (e.g., 'data_XX' where XX is a number).
    :return: Next available filename.
    """
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # List all files in the folder
    files = os.listdir(folder)

    # Extract numeric indices from filenames matching the pattern
    max_index = -1
    regex = re.compile(rf"{pattern}(\d{{2}})\..+")  # Matches 'data_XX.ext'

    for file in files:
        match = regex.match(file)
        if match:
            index = int(match.group(1))  # Extract the numeric part
            if index > max_index:
                max_index = index

    # Calculate the next index
    next_index = max_index + 1

    # Generate the next filename
    next_filename = f"{pattern}{next_index:02d}.{extension}"  # Adjust extension as needed
    return os.path.join(folder, next_filename)

def saveData(data, folder):
    print("Saving Data")
    new_file_path = find_next_filename(folder, "data_", "pkl")

    save_data = list(data)

    with open(new_file_path, "wb") as file:
        pickle.dump(save_data, file)

    print("Data Saved")