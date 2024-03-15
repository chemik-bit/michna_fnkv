import m2_cnn_pipeline as m2
import time

def run_model(config_path):
    start_time = time.time()
    try:
        m2.main(config_path)
    except Exception as e:
        print(f"Error training model with config {config_path}: {e}")
    end_time = time.time()
    return end_time - start_time

#first_time = run_model('src/cnn/configs/h_zkouska3.yaml')
second_time = run_model('src/cnn/configs/h_zkouska2.yaml')
third_time = run_model('src/cnn/configs/h_zkouska3.yaml')  # Assuming you want to try this again or it's a different config

#print(f"Execution time for first model: {first_time} seconds")
print(f"Execution time for second model: {second_time} seconds")
print(f"Execution time for third model: {third_time} seconds")
