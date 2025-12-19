import os
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from rondonia_trainer import Generator  # Import the brain structure we defined earlier

# --- CONFIGURATION ---
DATA_DIR = "data/ready_to_train"
MODEL_PATH = "data/rondonia_model_v1.pth"
OUTPUT_IMG = "rondonia_result_preview.png"
SAMPLES_TO_SHOW = 3

# Setup Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. Load the Model
    print(f"Loading model from {MODEL_PATH}...")
    model = Generator().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Switch to "Testing" mode (turns off learning)

    # 2. Get Random Samples
    hr_dir = os.path.join(DATA_DIR, "HR")
    lr_dir = os.path.join(DATA_DIR, "LR")
    all_files = os.listdir(hr_dir)
    
    selected_files = random.sample(all_files, SAMPLES_TO_SHOW)
    
    # 3. Setup the Plot
    # We create a grid: Rows = Samples, Cols = 3 (Input, AI Result, Truth)
    fig, axes = plt.subplots(SAMPLES_TO_SHOW, 3, figsize=(12, 4 * SAMPLES_TO_SHOW))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    transform = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    print("Running inference...")
    
    for i, filename in enumerate(selected_files):
        # Load Images
        hr_path = os.path.join(hr_dir, filename)
        lr_path = os.path.join(lr_dir, filename)
        
        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")
        
        # Prepare for AI (Convert to Tensor and add Batch dimension)
        lr_tensor = transform(lr_img).unsqueeze(0).to(DEVICE)
        
        # --- THE MAGIC HAPPENS HERE ---
        with torch.no_grad():
            fake_hr_tensor = model(lr_tensor)
        
        # Convert back to image for plotting
        fake_hr_img = to_pil(fake_hr_tensor.squeeze(0).cpu())
        
        # Plotting
        # Column 1: Low Res (Landsat)
        ax_lr = axes[i, 0] if SAMPLES_TO_SHOW > 1 else axes[0]
        ax_lr.imshow(lr_img)
        ax_lr.set_title("Input (Landsat 30m)", fontsize=10)
        ax_lr.axis("off")
        
        # Column 2: Super Res (AI Prediction)
        ax_sr = axes[i, 1] if SAMPLES_TO_SHOW > 1 else axes[1]
        ax_sr.imshow(fake_hr_img)
        ax_sr.set_title("AI Generated (SR-GAN)", fontsize=10, color="blue")
        ax_sr.axis("off")
        
        # Column 3: High Res (Sentinel Truth)
        ax_hr = axes[i, 2] if SAMPLES_TO_SHOW > 1 else axes[2]
        ax_hr.imshow(hr_img)
        ax_hr.set_title("Ground Truth (Sentinel 10m)", fontsize=10)
        ax_hr.axis("off")

    # Save the result
    plt.savefig(OUTPUT_IMG, bbox_inches='tight', dpi=150)
    print(f"--- SUCCESS! Preview image saved to {os.path.abspath(OUTPUT_IMG)} ---")

if __name__ == "__main__":
    main()
