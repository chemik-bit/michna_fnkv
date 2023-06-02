from pathlib import Path

path_to_models = Path(".")
models_list = []
for file in path_to_models.glob("*cnn*.py"):
    models_list.append(file.stem)
print(models_list)