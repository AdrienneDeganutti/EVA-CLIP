import torch
import cv2
import os
from shinji.eva_clip import create_model_and_transforms
from PIL import Image
import psutil
import numpy as np

def print_cpu_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"CPU memory used: {memory_info.rss / 1024 ** 2:.2f} MB") 

def print_gpu_memory():
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2 
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2 
        print(f"GPU memory allocated: {gpu_memory_allocated:.2f} MB")
        print(f"GPU memory reserved: {gpu_memory_reserved:.2f} MB")
    else:
        print("No GPU available.")

model_name = "EVA-CLIP-18B" 
pretrained = "/PATH/TO/PRETRAINED/MODEL/EVA_CLIP_18B_psz14_s6B.fp16.pt"    # UPDATE PATH TO YOUR PRETRAINED MODEL

device = torch.device("cuda")

print("Creating model and transforms...")
model, _, processor = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
print_cpu_memory()

print("Moving model to device...")
model = model.to(device)
print_gpu_memory()

# Load the video using OpenCV
video_directory = "/PATH/TO/VIDEO/FILES/DIRECTORY"    # UPDATE PATH TO THE DIRECOTRY CONTAINING THE VIDEO FILES
video_files = [f for f in os.listdir(video_directory)]

# Output directory for saving features
output_directory = "/PATH/TO/OUTPUT/DIRECTORY"        # UPDATE PATH TO OUTPUT THE FEATURES
os.makedirs(output_directory, exist_ok=True)  # Create the output directory if it doesn't exist


# Process each video file
for video_file in video_files:
    video_path = os.path.join(video_directory, video_file)
    print(f"Loading the video: {video_file}...")
    
    cap = cv2.VideoCapture(video_path)

    num_frames = 16                            # SET FRAME SUBSAMPLING VALUE
    print("Extracting frames...")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame indices, handling cases where total frames < num_frames separately
    if total_frames >= num_frames:
        step = total_frames / num_frames
        frame_indices = [int(i * step) for i in range(num_frames)]
    else:
        # If total frames are fewer than 16, use all available frames and duplicate as needed
        frame_indices = list(range(total_frames))

    frames = []
    print("Resizing frames...")
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert the frame (BGR to RGB) and resize it
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame).resize((224, 224))
            frames.append(pil_image)

    # If the total frames are fewer than 16, duplicate the last frame to reach 16
    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))

    cap.release()  

    # Preprocess the frames using EVA-CLIP's processor and stack them into a batch
    processed_frames = torch.stack([processor(frame) for frame in frames]).to(device)

    # Extract dense visual features
    print("Extracting features...")
    with torch.no_grad():
        image_features = model.encode_image(processed_frames)
        image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize the features

    print("Image Features Shape:", image_features.shape)

    # Save the features as a PyTorch tensor to the output directory
    feature_file_name = os.path.splitext(video_file)[0] + ".pt"
    feature_file_path = os.path.join(output_directory, feature_file_name)
    torch.save(image_features, feature_file_path)
    print(f"Features saved to '{feature_file_path}'.")
