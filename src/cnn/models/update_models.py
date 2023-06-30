from pathlib import Path

folder_to_process = Path(".","obsolete")
for file in folder_to_process.glob("*.py"):
    print(file)
    with open(file, "r") as f:
        filedata = f.read()
    filedata = filedata.replace("x.add(layers.Flatten())", "x.add(layers.GlobalMaxPooling2D())")

    with open(file.stem+"_gmp.py", "w") as fw:
        fw.write(filedata)
