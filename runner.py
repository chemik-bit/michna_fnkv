
from ga_steps import *
import m2_cnn_pipeline as m2
import time

num_generations = 2

for generation in range(num_generations):
    print("PUVODNI GENERACE", generation)
    binary_numbers_list = generate_individual(2, 136)

    for i, individual in enumerate(binary_numbers_list):
        model_creation(str(individual), model_index = i+1, generation = generation)
        print(f"Model {i+1} created")

    generation_runfile_creator(binary_numbers_list, generation = generation)

    #Path(__file__).parent.joinpath(f'./data/results/ga/{generation+1}').mkdir(parents=True, exist_ok=True)
    
    
    yaml_file = Path(__file__).parent.joinpath(f'./src/cnn/configs/ga/{generation+1}.yaml')
    m2.main(yaml_file, generation, ga = True)

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
    
