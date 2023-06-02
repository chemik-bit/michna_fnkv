from pathlib import Path

folder_to_process = Path(".")
for file in folder_to_process.glob("*.py"):
    print(file)
    with open(file, "r") as f:
        filedata = f.read()
    filedata = filedata.replace("x.add(layers.Flatten())", "x.add(layers.GlobalAveragePooling2D())")

    with open(file.stem+"_gap.py", "w") as fw:
        fw.write(filedata)
