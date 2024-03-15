
from ga_steps import *
import m2_cnn_pipeline as m2
import time

num_generations = 2

clear_directories()

for generation in range(num_generations):
    
    print("GENERACE NYNI: ", generation+1, "\n\n\n\n")
    if generation == 0:
        binary_numbers_list = generate_individual(3, 136)
    else:
        new_binary_numbers_list = generate_individual(5, 136)
        binary_numbers_list = binary_numbers_list + new_binary_numbers_list


    Path(__file__).parent.joinpath(f'./data/results/ga/{generation+1}').mkdir(parents=True, exist_ok=True)
    with open(Path(__file__).parent.joinpath(f'./data/results/ga/{generation+1}/individuals.txt'), 'w') as file:
        file.write(str(binary_numbers_list))
    
    for i, individual in enumerate(binary_numbers_list):
        model_creation(str(individual), model_index = i+1, generation = generation)
        print(f"Model {i+1} created")

    generation_runfile_creator(binary_numbers_list, generation = generation)
    
    yaml_file = Path(__file__).parent.joinpath(f'./src/cnn/configs/ga/{generation+1}.yaml')
    m2.main(yaml_file, generation, ga = True)
    
    
    selected_individuals, selected_indexes = read_and_sort_results(generation, 5)
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
    
