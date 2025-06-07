from PIL import Image, ImageOps
import os

input_dir = "archive/Training"
output_dir = "archive_augmented/Training"

def augment_images(input_dir, output_dir):
    classes = os.listdir(input_dir)
    for cls in classes:
        input_cls_path = os.path.join(input_dir, cls)
        output_cls_path = os.path.join(output_dir, cls)
        os.makedirs(output_cls_path, exist_ok=True)

        for fname in os.listdir(input_cls_path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            original_path = os.path.join(input_cls_path, fname)
            image = Image.open(original_path).convert('L')

            # Speichere Originalbild
            image.save(os.path.join(output_cls_path, fname))

            # Horizontal Flip
            flipped_h = ImageOps.mirror(image)
            flipped_h.save(os.path.join(output_cls_path, f"{fname.split('.')[0]}_flipH.png"))

            # Vertical Flip
            flipped_v = ImageOps.flip(image)
            flipped_v.save(os.path.join(output_cls_path, f"{fname.split('.')[0]}_flipV.png"))

augment_images(input_dir, output_dir)
