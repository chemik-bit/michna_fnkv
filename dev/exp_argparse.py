import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configfile", type=Path, required=True)
    args = parser.parse_args()
    with open(args.configfile, "r", encoding="UTF-8") as f:
        print(f.readlines())
