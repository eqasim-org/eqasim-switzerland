import os
import xml.etree.ElementTree as ET
from xmldiff import main
import logging
logger = logging.getLogger("synpp")

def scan_for_use_vdf(directory, item="use_vdf"):
    """
    Scans all Python (.py) files in the given directory and its subdirectories
    to find those containing the string item, excluding the current script.

    Args:
        directory (str): The path to the directory to scan.

    Returns:
        list: A list of file paths that contain the string item.
    """
    matching_files = []

    # Get the absolute path of the current script
    current_script_path = os.path.abspath(__file__)

    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file is a Python file
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                file_abs_path = os.path.abspath(file_path)

                # Skip the current script
                if file_abs_path == current_script_path:
                    continue

                # Open and read the file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Check if the string item is in the file
                    if item in content:
                        matching_files.append(file_path)
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")

    return matching_files



# Example usage
if __name__ == "__main__":
    # Specify the directory to scan
    directory_to_scan = os.getcwd()
    item = "matsim.scenario.vehicles"

    logger.info(f"Scanning {directory_to_scan}")
    logger.info(f"Searching for '{item}'")

    # Get the list of files containing "use_vdf"
    files_with_use_vdf = scan_for_use_vdf(directory_to_scan, item)
    
    # Print the results
    if files_with_use_vdf:
        logger.info(f"Files containing '{item}':")
        for file in files_with_use_vdf:
            logger.info(file)
    else:
        logger.info("No files containing 'use_vdf' were found.")
