
from ga_steps import *
import m2_cnn_pipeline as m2
import time

num_generations = 10

clear_directories()

for generation in range(num_generations):
    
    
    print("GENERACE NYNI: ", generation+1, "\n\n\n\n")
    if generation == 0:
        binary_numbers_list = generate_individual(200, 220)
    else:
        new_binary_numbers_list = generate_individual(140, 220)
        binary_numbers_list = binary_numbers_list + new_binary_numbers_list


    Path(__file__).parent.joinpath(f'./data/results/ga/{generation+1}').mkdir(parents=True, exist_ok=True)
    with open(Path(__file__).parent.joinpath(f'./data/results/ga/{generation+1}/individuals.txt'), 'w') as file:
        file.write(str(binary_numbers_list))
    
    for i, individual in enumerate(binary_numbers_list):
        model_creation(str(individual), model_index = i+1, generation = generation)
        print(f"Model {i+1} created")
    
    generation_runfile_creator(binary_numbers_list, generation = generation)
    for individual in range(len(binary_numbers_list)):
        yaml_file = Path(__file__).parent.joinpath(f'./src/cnn/configs/ga/{generation+1}/config_{individual+1}.yaml')
        print("YAML FILE: ", yaml_file)
        m2.main(yaml_file, generation, individual, ga = True)
    
    
    
    selected_individuals, selected_indexes = read_and_sort_results(generation, 30)
    """
    for result in selected_individuals:
        print(f"Filename: {result[0]}, Validation Accuracy: {result[1]}")
    """
    binary_numbers_list = crossover(selected_individuals, selected_indexes, generation)
    
    











"""
def run_model(config_path):
    start_time = time.time()
    try:
        m2.main(config_path)
    except Exception as e:
        print(f"Error training model with config {config_path}: {e}")
    end_time = time.time()
    return end_time - start_time
"""
    
