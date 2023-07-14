from pathlib import Path

folder_to_process = Path(".","2conv_2x4_best")
for file in folder_to_process.glob("*.py"):
    print(file)
    with open(file, "r") as f:
        filedata = f.read()
    filedata = filedata.replace("x.add(layers.Dropout(", "x.add(layers.Dropout(0.")

    with open(file, "w") as fw:
        fw.write(filedata)
