import os

import torch
import torchmetrics

from torch import nn, optim
from tqdm import tqdm

from dragonfruitvp.data.custom_dataset import CompetitionDataset
from dragonfruitvp.src.unet import UNet


if __name__ == "__main__":
    BATCH_SIZE = 8
    limit = None
    # base_datadir = '/scratch/yg2709/CSCI-GA-2572-Deep-Learning-Final-Competition-Dragonfruit/dataset'
    base_datadir = './dataset'
    train_set = CompetitionDataset(os.path.join(base_datadir, 'train'), dataset_type='labeled', limit=limit) # we treat trainset as unlabeled here
    val_set = CompetitionDataset(os.path.join(base_datadir, 'val'), dataset_type='labeled', limit=limit)
    
    num_workers = 2
    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers
    )
    dataloader_val = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers
    )
    
    
    model = UNet() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49).to(device)
    
    num_epochs = 50
    best_iou = 0.0  # Initialize best IoU
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_iou = 0
        for pre_seqs, aft_seqs, masks in tqdm(dataloader_train):
            images = torch.cat((pre_seqs, aft_seqs), dim=1)
            # print(images.shape, masks.shape)
            B, T, C, H, W = images.shape
            images = images.view(B*T, C, H, W)
            masks = masks.view(B*T, H, W)

            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_iou += jaccard(outputs, masks).item()
            
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)  

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')


        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_pre_seqs, val_aft_seqs, val_masks in tqdm(val_loader):
                val_images = torch.cat((val_pre_seqs, val_aft_seqs), dim=1)
                B, T, C, H, W = val_images.shape
                val_images = val_images.view(B*T, C, H, W)
                val_masks = val_masks.view(B*T, H, W)

                val_images, val_masks = val_images.to(device), val_masks.to(device)
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_masks).item()
                
                # 计算 IoU
        val_loss /= len(val_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}')
        
        # Save the best model if the current IoU is greater than the best recorded IoU
        save_dir = 'weights_hub/unet'
        os.makedirs(saved_dir, exist_ok=True)

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'best_model.pth')
        print(f'Best model saved with IoU: {best_iou:.4f}')