import os
from PIL import Image
from utils.args_config_inference import parse_args

def main():
    args = parse_args()

    # 1. Load and print contents of example.txt in embed_dir
    txt_path = os.path.join(args.embed_dir, "example.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            content = f.read()
        print(f"Contents of example.txt:\n{content}")
    else:
        print(f"No example.txt found in {args.embed_dir}")

    # 2. Create output_dir and save a dummy image
    os.makedirs(args.output_dir, exist_ok=True)
    img = Image.new("RGB", (100, 100), color="white")
    image_path = os.path.join(args.output_dir, "dummy_image.png")
    img.save(image_path)
    print(f"Dummy image saved at: {image_path}")

if __name__ == "__main__":
    main()
