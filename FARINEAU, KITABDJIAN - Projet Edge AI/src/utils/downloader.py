import requests
import os


def create_directory(directory):
    """
    Create a directory if it does not exist.

    Args:
        directory (str): The path of the directory to be created.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_file(link, directory, filename):
    """
    Download a file from a link and save it to a specified directory.

    Args:
        link (str): The URL of the file to be downloaded.
        directory (str): The directory where the file will be saved.
        filename (str): The name of the saved file.
    """
    target_file = os.path.join(directory, filename)

    if os.path.exists(target_file):
        print("File exists, skipping")
        return

    create_directory(directory)

    if "/" in target_file:
        target_file = target_file.replace("/", "_")
    if "?" in target_file:
        target_file = target_file.replace("?", "_")

    with open(target_file, 'wb') as f:
        f.write(requests.get(link).content)


def get_bird_json(bird_name, recording_quality, number_of_recordings):
    """
    Fetch bird recordings information in JSON format from the xeno-canto API.

    Args:
        bird_name (str): The name of the bird to fetch recordings for.
        recording_quality (list): A list of recording qualities to fetch.
        number_of_recordings (int): The maximum amount recordings to return.

    Returns:
        list: A list of JSON objects containing bird recordings information.
    """
    api_base_link = "https://xeno-canto.org/api/2/recordings?query="
    quality_header = "q:"
    bird_name = "+".join(bird_name.split(" "))

    final_json = []

    for quality in recording_quality:
        print("Getting recordings data for", bird_name, "with quality", quality)
        api_link = api_base_link + bird_name + "+" + quality_header + quality
        json_recording = requests.get(api_link).json()["recordings"]
        final_json.extend(json_recording)

    final_json = [recording for recording in final_json if "song" in recording["type"]]
    return final_json[:number_of_recordings]


def download_from_bird_json_infos(recordings_folder, bird_json):
    """
    Download bird recordings from the JSON information.

    Args:
        recordings_folder (str): The directory to save the downloaded recordings.
        bird_json (list): A list of JSON objects containing bird recordings information.
    """
    len_bird_recordings = len(bird_json)

    for index, recording in enumerate(bird_json):
        print("Downloading file:", index + 1, "out of", len_bird_recordings, "(", recording["file-name"], ")")
        bird_folder = os.path.join(recordings_folder, recording["gen"] + "_" + recording["sp"])
        filename = recording["q"] + "_" + recording["file-name"]
        download_file(recording["file"], bird_folder, filename)
