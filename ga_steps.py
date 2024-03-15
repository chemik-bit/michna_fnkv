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
            num_conv_layers = max(int(binary[:3], 2), 1)
            for conv_layer in range(num_conv_layers):
                setup = binary[3 + conv_layer * 15: 3 + (conv_layer + 1) * 15]
                #print("setup", setup)
                conv_filters = int(setup[:6], 2)
                filter_size = max(int(setup[6:9], 2), 1)
                activation = {0: "sigmoid", 1: "tanh", 2: "relu", 3: "leaky_relu"}[int(setup[9:10], 2)]
                if conv_layer == 0:
                    file.write(f"    x.add(layers.Conv2D({conv_filters}, ({filter_size}, {filter_size}), activation='{activation}', padding='same', input_shape=(input_size[0], input_size[1], 1)))\n")
                else:
                    file.write(f"    x.add(layers.Conv2D({conv_filters}, ({filter_size}, {filter_size}), activation='{activation}', padding='same'))\n")
                max_pooling = int(setup[11:12], 2)
                pool_size = max(int(setup[12:15], 2),1)
                if max_pooling == 1:
                    file.write(f"    x.add(layers.MaxPooling2D(pool_size=({pool_size}, {pool_size}), strides=(2, 2)))\n")

            file.write("    x.add(layers.GlobalAveragePooling2D())\n")

            num_dense_layers = max(int(binary[48:50], 2), 1)
            for dense_layer in range(num_dense_layers):
                setup = binary[50 + dense_layer * 13: 50 + (dense_layer + 1) * 13]
                dense_neurons = max(int(setup[:11], 2), 10)
                activation = {0: "sigmoid", 1: "tanh", 2: "relu", 3: "leaky_relu"}[int(setup[11:], 2)]
                file.write(f"    x.add(layers.Dense({dense_neurons}, activation='{activation}'))\n")
            
            file.write("    x.add(layers.Dense(1, activation=\"sigmoid\"))\n")
            file.write("    return x")

def generation_runfile_creator(binary_numbers_list, generation):
    Path(__file__).parent.joinpath(f'./src/cnn/configs/ga').mkdir(parents=True, exist_ok=True)
    models_list = ',\n        '.join([f"'src.cnn.models.ga.{generation+1}.model_{i+1}'" for i in range(len(binary_numbers_list))])
    binary_numbers_str = ',\n        '.join([f"'{binary}'" for binary in binary_numbers_list])
            
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
batch_size_exp: 32
max_epochs: 500
lr: 0.1
transform: v1
models: [
        {models_list}
        ]
loss: focal_loss
optimizer: adam
focal_loss_gamma: 2
binary: [
        {binary_numbers_str}
        ]
"""         
            
    with open(f"src/cnn/configs/ga/{generation+1}.yaml", "w") as new_yaml_file:
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

    top_binaries = [result[2] for result in results[:5]]
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