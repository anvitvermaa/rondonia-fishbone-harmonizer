import os
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import warnings
import sys

# Import Generator from the trainer script
sys.path.append('scripts')
try:
    from rondonia_trainer import Generator
except ImportError:
    # Fallback if running from inside scripts/
    try:
        from scripts.rondonia_trainer import Generator
    except ImportError:
        # Final fallback - assume it's in the same dir
        import rondonia_trainer
        from rondonia_trainer import Generator

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATA_DIR = "data/ready_to_train"
MODEL_PATH = "data/rondonia_model_v1.pth"
MAX_SAMPLES = 100 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_sam(pred, target):
    """Calculates Spectral Angle Mapper (SAM) in radians"""
    h, w, c = pred.shape
    pred_flat = pred.reshape(-1, c)
    target_flat = target.reshape(-1, c)
    pred_norm = np.maximum(np.linalg.norm(pred_flat, axis=1), 1e-8)
    target_norm = np.maximum(np.linalg.norm(target_flat, axis=1), 1e-8)
    dot = np.sum(pred_flat * target_flat, axis=1)
    cos_theta = np.clip(dot / (pred_norm * target_norm), -1, 1)
    return np.mean(np.arccos(cos_theta))

def calculate_metrics():
    print(f"--- Loading Model for Evaluation ---")
    model = Generator().to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()
    
    hr_dir = os.path.join(DATA_DIR, "HR")
    lr_dir = os.path.join(DATA_DIR, "LR")
    
    if not os.path.exists(hr_dir):
        print(f"Error: Data directory {hr_dir} not found.")
        return
        
    files = sorted(os.listdir(hr_dir))[:MAX_SAMPLES]
    transform = transforms.ToTensor()
    
    total_psnr, total_ssim, total_sam = 0.0, 0.0, 0.0
    valid_count = 0
    
    print(f"Calculating metrics on {len(files)} samples...")
    
    with torch.no_grad():
        for filename in tqdm(files):
            try:
                hr_path = os.path.join(hr_dir, filename)
                lr_path = os.path.join(lr_dir, filename)
                
                hr_img = Image.open(hr_path).convert("RGB")
                lr_img = Image.open(lr_path).convert("RGB")
                
                lr_tensor = transform(lr_img).unsqueeze(0).to(DEVICE)
                fake_hr_tensor = model(lr_tensor)
                
                fake_hr_np = fake_hr_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
                real_hr_np = transform(hr_img).permute(1, 2, 0).numpy()
                
                total_psnr += psnr(real_hr_np, fake_hr_np, data_range=1.0)
                total_ssim += ssim(real_hr_np, fake_hr_np, data_range=1.0, channel_axis=2)
                total_sam += calculate_sam(fake_hr_np, real_hr_np)
                valid_count += 1
            except Exception:
                continue

    if valid_count == 0:
        print("No valid images processed.")
        return

    print(f"\n--- REPORT CARD (Pilot Phase - 3 Epochs) ---")
    print(f"Samples Tested: {valid_count}")
    print(f"PSNR (Accuracy):  {total_psnr/valid_count:.2f} dB  [Goal: >28.0]")
    print(f"SSIM (Texture):   {total_ssim/valid_count:.4f}     [Goal: >0.80]")
    print(f"SAM (Color Phys): {total_sam/valid_count:.4f} rad  [Goal: <0.10]")
    print(f"--------------------------------------------")
    if (total_ssim/valid_count) > 0.60:
        print("Verdict: ARCHITECTURE VERIFIED. Ready for Gaming Lab.")
    else:
        print("Verdict: Architecture needs review (SSIM too low).")

if __name__ == "__main__":
    calculate_metrics()
