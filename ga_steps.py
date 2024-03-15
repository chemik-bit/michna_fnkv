import random
import shutil

def clear_directories():
    directories = ['./data/results/ga', './src/cnn/configs/ga', './src/cnn/models/ga']
    for directory in directories:
        try:
            shutil.rmtree(Path(__file__).parent.joinpath(directory))
            print(f"Directory {directory} cleared.")
        except FileNotFoundError:
            print(f"Directory {directory} does not exist, skipping.")

def generate_individual(count, length):
    binary_numbers = []
    for _ in range(count):
        number = ''.join(str(random.randint(0, 1)) for _ in range(length))
        binary_numbers.append(number)
    
    return binary_numbers

from pathlib import Path

def model_creation(binary: str, model_index: int, generation: int):
    #Path(__file__).parent.joinpath('./src/cnn/models/ga').mkdir(parents=True, exist_ok=True)
    Path(__file__).parent.joinpath(f'./src/cnn/models/ga/{generation+1}').mkdir(parents=True, exist_ok=True)
    with open(Path(__file__).parent.joinpath(f'./src/cnn/models/ga/{generation+1}/model_{model_index}.py'), 'w') as file:
            file.write("from tensorflow.keras import layers\nimport tensorflow as tf\n\n\ndef create_model(input_size):\n    x = tf.keras.Sequential()\n")
            intro = 24 
            num_conv_layers = max(int(binary[intro:intro+3], 2), 1)
            conv_layer_length = 18
            for conv_layer in range(num_conv_layers):
                setup = binary[intro + 3 + conv_layer * conv_layer_length: intro + 3 + (conv_layer + 1) * conv_layer_length]                #print("setup", setup)
                conv_filters = max(int(setup[:6], 2),1)
                filter_size = max(int(setup[6:9], 2), 1)
                activation = {0: "sigmoid", 1: "tanh", 2: "relu", 3: "leaky_relu"}[int(setup[9:11], 2)]
                if conv_layer == 0:
                    file.write(f"    x.add(layers.Conv2D({conv_filters}, ({filter_size}, {filter_size}), activation='{activation}', padding='same', input_shape=(input_size[0], input_size[1], 1)))\n")
                else:
                    file.write(f"    x.add(layers.Conv2D({conv_filters}, ({filter_size}, {filter_size}), activation='{activation}', padding='same'))\n")
                if setup[11] == "1":
                    file.write(f"    x.add(layers.BatchNormalization())\n")
                pool_size = max(int(setup[13:15], 2),1)
                strides = max(int(setup[15:17], 2),1)
                if setup[12] == "1":
                    file.write(f"    x.add(layers.MaxPooling2D(pool_size=({pool_size}, {pool_size}), strides=(2, 2)))\n")
                if setup[17] == "1":
                    file.write(f"    x.add(layers.BatchNormalization())\n")
            file.write("    x.add(layers.GlobalAveragePooling2D())\n")
            conv_binary_section = 129
            dense_layer_length = 17
            num_dense_layers = max(int(binary[intro + conv_binary_section:intro + conv_binary_section + 2], 2), 1)
            for dense_layer in range(num_dense_layers):
                setup = binary[intro + conv_binary_section + 2 + dense_layer * dense_layer_length: intro + conv_binary_section + 2 + (dense_layer + 1) * dense_layer_length]
                dense_neurons = max(int(setup[:11], 2), 10)
                activation = {0: "sigmoid", 1: "tanh", 2: "relu", 3: "leaky_relu"}[int(setup[11:13], 2)]
                file.write(f"    x.add(layers.Dense({dense_neurons}, activation='{activation}'))\n")
                dropout_mapping = {"000": 0.2, "001": 0.3, "010": 0.4, "011": 0.5, "100": 0.6, "101": 0.7, "110": 0.8, "111": 0.9}
                percentage = dropout_mapping[setup[14:]]
                if setup[13] == "1":
                    file.write(f"    x.add(layers.Dropout({percentage}))\n")
            
            file.write("    x.add(layers.Dense(1, activation=\"sigmoid\"))\n")
            file.write("    return x")

def generation_runfile_creator(binary_numbers_list, generation):
    Path(__file__).parent.joinpath(f'./src/cnn/configs/ga/{generation+1}').mkdir(parents=True, exist_ok=True)
    for individual in range(len(binary_numbers_list)):
        model = f"'src.cnn.models.ga.{generation+1}.model_{individual+1}'"
        binary = binary_numbers_list[individual]
        batch_size_exp = max(int(binary[:7], 2), 1)
        max_epochs = max(int(binary[8:17], 2)+25, 25)
        lr = {0: 0.01, 1: 0.001, 2: 0.0001, 3: 0.00001}[int(binary[17:19], 2)]
        transform_version = int(binary[19], 2)
        transform = f"v{transform_version + 1}"
        loss_choice = "binary_crossentropy" if int(binary[20], 2) == 0 else "focal_loss"
        optimizer_choice = "adam" if int(binary[21:23], 2) == 0 else "adagrad" if int(binary[21:23], 2) == 1 else "sgd" if int(binary[21:23], 2) == 2 else "rmsprop"
        gamma = int(binary[23:25], 2) + 1

        new_yaml_content = f"""
    image_size:
        1: [60, 240]
    balances: [True]
    wav_chunks: [1]
    octaves: []
    fft_lens: [1250]
    fft_overlaps: [625]
    training_db: svdadult
    validation_db: svdadult
    batch_size_exp: {batch_size_exp}
    max_epochs: {max_epochs}
    lr: {lr}
    transform: {transform}
    models: [
            {model}
            ]
    loss: {loss_choice}
    optimizer: {optimizer_choice}
    focal_loss_gamma: {gamma}
    binary: [
            '{binary}'
            ]
    """         
                
        with open(f"src/cnn/configs/ga/{generation+1}/config_{individual+1}.yaml", "w") as new_yaml_file:
            new_yaml_file.write(new_yaml_content)

import os
import json

def read_and_sort_results(generation, top_n=5):
    results_path = Path(__file__).parent.joinpath(f'./data/results/ga/{generation+1}')
    results = []
    for filename in os.listdir(results_path):
        if filename.endswith('.json'):
            with open(os.path.join(results_path, filename), 'r') as file:
                data = json.load(file)
                if 'val_acc' in data:
                    number = filename.split('.')[0]
                    results.append((number, data['val_acc'], data["binary"]))
    results.sort(key=lambda x: x[1], reverse=True)

    sorted_file_path = Path(__file__).parent.joinpath(f'./data/results/ga/{generation+1}/sorted.txt')
    with open(sorted_file_path, 'w') as file:
        for result in results:
            file.write(f"Filename: {result[0]}, Validation Accuracy: {result[1]}, Binary: {result[2]}\n")
    #print("vysledky", results)
    
    """
    individuals_path = Path(__file__).parent.joinpath(f'./data/results/ga/{generation+1}/individuals.txt')
    with open(individuals_path, 'r') as file:
        binary_numbers_list = eval(file.read())
    """

    top_binaries = [result[2][0] for result in results[:5]]
    top_indexes = [result[0] for result in results[:5]]
    return top_binaries, top_indexes

import random

def crossover(binary_numbers_list, bin_indexes, generation):
    # Duplicate the list of binary numbers
    duplicated_list = binary_numbers_list.copy()
    crossover_results = binary_numbers_list.copy()
    
    # Iterate over the duplicated list in pairs
    for i in range(0, len(duplicated_list)):
        # Select two binary numbers for crossover
        binary1 = duplicated_list[i]

        #binary2 = random.choice(duplicated_list)
        binary2_index = random.randint(0, len(duplicated_list) - 1)
        binary2 = duplicated_list[binary2_index]

        binary3_index = random.randint(0, len(duplicated_list) - 1)
        binary3 = duplicated_list[binary3_index]

        #print(f"Index of orig binary: {bin_indexes[i]}")
        #print(f"Index of swap binary: {binary2_index+1}")
        #print(f"Index of swap binary: {binary3_index+1}")
        
        # Pick a random length for the cut
        cut_length = random.randint(1, len(binary1))
        cut_length2 = random.randint(1, len(binary1))
        #print("cut", cut_length)
        
        # Perform the crossover
        new_binary1 = binary1[:cut_length] + binary2[cut_length:]
        new_binary2 = binary1[:cut_length2] + binary3[cut_length2:]
        
        # Add the new binary numbers to the results list
        crossover_results.extend([new_binary1, new_binary2])

        sorted_file_path = Path(__file__).parent.joinpath(f'./data/results/ga/{generation+1}/sorted.txt')
        with open(sorted_file_path, 'a') as file:
            file.write(f"({binary2_index}, {cut_length}), ({binary3_index}, {cut_length2})\n")

        #print("delka vysledky", len(crossover_results))
        #print("delka vstupni", len(duplicated_list))
    
        #print("original ", duplicated_list[i])
        #print("cut      ", binary1[:cut_length])
        #print("crossover", crossover_results[-1])

        #print("\n\n\n")
        
        #print("original ", duplicated_list[i])
        #print("cut      ", binary1[:cut_length])
        #print("crossover", crossover_results[-2])
    
    return crossover_results