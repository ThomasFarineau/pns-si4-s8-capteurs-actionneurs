import os
import wave

import numpy as np
import tensorflow as tf
from keras import Sequential, Input
from keras.layers import MaxPool1D, Conv1D, AvgPool1D, Flatten, Activation, Dense
from keras.utils import to_categorical


def get_sound_files(directory):
    """
    Get sound files from the specified folder

    Args:
        directory (str): The folder containing bird recordings.

    Returns:
        list: A list of sound file paths.
    """
    sound_files = []

    for bird_type in os.listdir(directory):
        bird_folder = os.path.join(directory, bird_type)
        if os.path.isdir(bird_folder):
            for recording in os.listdir(bird_folder):
                sound_filename = os.path.join(bird_folder, recording)
                if os.path.isfile(sound_filename) and "splitted_" in sound_filename:
                    sound_files.append(sound_filename)

    return sound_files


def create_test_file(directory, bird, file):
    """
    Create a test file containing a list of sound file paths for testing.

    Args:
        directory (str): The folder containing bird recordings.
        bird (str): The main bird to be considered.
        file (str): The name of the output test file.
    """
    sound_files = get_sound_files(directory)
    count = 0

    with open(os.path.join(directory, file), "w", encoding="utf-8") as f:
        for sound_file in sound_files:
            if np.random.rand() > 0.3:
                f.write(sound_file + "\n")
                count += 1

        ratio = count / len(os.listdir(os.path.join(directory, bird.replace(" ", "_"))))
        number_of_main_bird_recordings = count / (
                sum(os.path.isdir(os.path.join(directory, f)) for f in os.listdir(directory)) - 1)

        count = 0
        bird_folder = os.path.join(directory, bird.replace(" ", "_"))

        for recording in os.listdir(bird_folder):
            if count > number_of_main_bird_recordings:
                break

            # If random number between 0 and 1 is greater than 0.3, then add to testing_list.txt
            if np.random.rand() > ratio:
                count += 1
                f.write(os.path.join(bird_folder, recording) + "\n")


def read_test_list(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        testing_list = f.read().splitlines()
    return testing_list


def process_audio_files(directory, bird_classes, testing_list):
    x_train, y_train, x_test, y_test = [], [], [], []
    for bird_type in os.listdir(directory):
        bird_folder = os.path.join(directory, bird_type)
        if os.path.isdir(bird_folder):
            for recording in os.listdir(bird_folder):
                if "splitted_" not in recording:
                    continue
                if bird_type not in bird_classes:
                    continue
                sound_filename = os.path.join(bird_folder, recording)
                label = bird_classes.index(bird_type)

                with wave.open(sound_filename) as f:
                    data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).copy()

                data = data.astype(np.float32)
                data.resize((16000, 1))

                if sound_filename in testing_list:
                    x_test.append(data)
                    y_test.append(label)
                elif y_train.count(label) < 2400:
                    x_train.append(data)
                    y_train.append(label)

    return x_train, y_train, x_test, y_test


def normalize_data(x_train, x_test):
    x_mean = x_train.mean()
    x_std = x_train.std()

    x_train -= x_mean
    x_test -= x_mean
    x_train /= x_std
    x_test /= x_std

    return x_train, x_test


def create_dataset(directory, bird_classes, test_file):
    testing_list = read_test_list(os.path.join(directory, test_file))
    x_train, y_train, x_test, y_test = process_audio_files(directory, bird_classes, testing_list)

    x_train, y_train, x_test, y_test = np.array(x_train), to_categorical(np.array(y_train)), np.array(
        x_test), to_categorical(np.array(y_test))

    x_train, x_test = normalize_data(x_train, x_test)

    return x_train, y_train, x_test, y_test


def build_model():
    model = Sequential()
    model.add(Input(shape=(16000, 1)))
    model.add(MaxPool1D(pool_size=20, padding='valid'))
    model.add(Conv1D(filters=8, kernel_size=40, activation='relu'))
    model.add(MaxPool1D(pool_size=4, padding='valid'))
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(MaxPool1D(pool_size=4, padding='valid'))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPool1D(pool_size=4, padding='valid'))
    model.add(AvgPool1D(pool_size=8))
    model.add(Flatten())
    model.add(Dense(units=3))
    model.add(Activation('softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=10e-4)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


def create_model(directory, bird_classes, test_file):
    x_train, y_train, x_test, y_test = create_dataset(directory, bird_classes, test_file)

    np.savetxt('x_test.csv', x_test.reshape(x_test.shape[0], -1), delimiter=',', fmt='%s')
    np.savetxt('y_test.csv', y_test, delimiter=',', fmt='%s')

    model = build_model()
    model.summary()

    # Train Model
    model.fit(x_train, y_train, epochs=5, batch_size=100, validation_data=(x_test, y_test))

    # Evaluate model on test dataset
    model.evaluate(x_test, y_test, verbose=2)
    pred_test = model.predict(x_test)
    print(tf.math.confusion_matrix(y_test.argmax(axis=1), pred_test.argmax(axis=1)))

    # Save model
    model.save('model.h5')

    # Remove the Softmax layer
    model = tf.keras.Model(model.input, model.layers[-2].output, name=model.name)
    return model
