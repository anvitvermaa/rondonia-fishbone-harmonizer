import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = "data/ready_to_train"
MODEL_SAVE_PATH = "data/rondonia_model_v1.pth"
BATCH_SIZE = 8          # Keep small for laptop/pilot testing
EPOCHS = 3              # Just a few epochs to prove it learns
LR_RATE = 0.0001        # Learning Rate

# Check for GPU (CUDA) or use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Training on: {DEVICE} ---")

# --- 1. THE DATASET LOADER ---
class FishboneDataset(Dataset):
    def __init__(self, root_dir):
        self.hr_dir = os.path.join(root_dir, "HR")
        self.lr_dir = os.path.join(root_dir, "LR")
        self.image_names = os.listdir(self.hr_dir)
        
        # Standardize images to Tensor (0-1 range)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        # Load High Res (Target) and Low Res (Input)
        hr_img = Image.open(os.path.join(self.hr_dir, img_name)).convert("RGB")
        lr_img = Image.open(os.path.join(self.lr_dir, img_name)).convert("RGB")
        
        return self.transform(lr_img), self.transform(hr_img)

# --- 2. THE MODEL (Simple SRResNet) ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual

class Generator(nn.Module):
    def __init__(self, scale_factor=1):
        # NOTE: Since we already aligned/upscaled Landsat in the 'Processor' step
        # via reprojection, our Input and Output are the SAME size (256x256).
        # The AI just needs to "sharpen" (Refine) the image, not upscale dimensions.
        super(Generator, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()
        
        # 5 Residual Blocks for the "Thinking" part
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(5)])
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4)
        self.sigmoid = nn.Sigmoid() # Output 0-1 pixel values

    def forward(self, x):
        block1 = self.prelu(self.conv1(x))
        blocks = self.res_blocks(block1)
        block2 = self.bn2(self.conv2(blocks))
        output = self.conv3(block1 + block2)
        return self.sigmoid(output)

# --- 3. TRAINING LOOP ---
def train():
    # Setup Data
    dataset = FishboneDataset(DATA_DIR)
    # num_workers=0 to avoid bugs with external drives on Linux pilots
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
    
    # Setup Model
    model = Generator().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
    criterion = nn.MSELoss() # Simple Mean Squared Error loss for the pilot

    print(f"Loaded {len(dataset)} training chips.")
    print("Starting Training Loop...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        # Progress Bar
        loop = tqdm(dataloader, leave=True)
        
        for idx, (lr, hr) in enumerate(loop):
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            
            # Forward Pass (The AI Guesses)
            fake_hr = model(lr)
            
            # Loss Calculation (How wrong was it?)
            loss = criterion(fake_hr, hr)
            
            # Backward Pass (The AI Learns)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Completed. Avg Loss: {epoch_loss / len(dataloader):.6f}")

    # Save the Brain
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n--- SUCCESS! Model saved to {MODEL_SAVE_PATH} ---")

if __name__ == "__main__":
    train()
