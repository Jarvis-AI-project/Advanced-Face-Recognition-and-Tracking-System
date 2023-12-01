import os

# List all images inside a folder
def list_images(path):
    image_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    image_files.sort()
    return image_files
