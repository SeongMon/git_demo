import os
from utils.args_config_train import parse_args

def main():
    args = parse_args()
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Create a simple text file in the output_dir
    file_path = os.path.join(args.output_dir, "example.txt")
    with open(file_path, "w") as f:
        f.write("demo text file for training script.\n")

    print(f"Text file created at: {file_path}")

if __name__ == "__main__":
    main()
