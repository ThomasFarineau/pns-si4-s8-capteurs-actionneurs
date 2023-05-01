import os
from pathlib import Path
import copy
from utils.audio import split_audio_file
from utils.downloader import get_bird_json, download_from_bird_json_infos
from utils.training import create_test_file, create_model
import kerascnn2c


def main():
    folder, file = "recordings", "testing_list.txt"
    birds, classes = [['Fringilla coelebs', ['A', 'B', 'C'], 300], ["Sterna hirundo", ["A", "B", "C"], 200],
                      ["Sylvia atricapilla", ["A", "B", "C"], 200]], []

    for bird in birds:
        json = get_bird_json(bird[0], bird[1], bird[2])
        download_from_bird_json_infos(folder, json)
        classes.append(bird[0].replace(" ", "_"))

    print("Splitting audio into multiple recordings")

    # Iterate over bird types in the recordings folder
    for bird_type in os.listdir(folder):
        bird_folder = os.path.join(folder, bird_type)
        print("Looking into folder:", bird_folder)

        # Check if the current item is a directory
        if os.path.isdir(bird_folder):
            number_of_recordings = len(os.listdir(bird_folder))

            # Iterate over the recordings in the bird folder
            for index, recording in enumerate(os.listdir(bird_folder)):
                print(
                    f"Splitting recording {index + 1} ({recording}) out of {number_of_recordings} for bird {bird_type}")
                recording_path = os.path.join(bird_folder, recording)

                # Split the audio file into 3-second segments
                split_audio_file(recording_path, bird_folder, recording.split(".")[0], 3)

    # Create the test file
    create_test_file(folder, birds[0][0], file)

    # create model
    model = create_model(folder, classes, file)

    # generate c code
    res = kerascnn2c.Converter(output_path=Path('gsc_output'), fixed_point=9, number_type='int16_t',
                               long_number_type='int32_t', number_min=-(2 ** 15),
                               number_max=(2 ** 15) - 1).convert_model(copy.deepcopy(model))
    with open('src/utils/gsc_model.h', 'w') as f:
        f.write(res)


if __name__ == "__main__":
    main()
