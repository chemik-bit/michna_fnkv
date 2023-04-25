import yaml

# read yaml file
with open("cnn_pipeline_config.yaml") as file:
    yaml_data = yaml.safe_load(file)


for key in yaml_data.keys():
    print(f"SECTION -------------------------------------- {key}: {yaml_data[key]}")
