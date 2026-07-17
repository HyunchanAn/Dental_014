import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class OsteoporosisDataset(Dataset):
    """
    Dataset for Osteoporosis Risk Screening (Sharda Dataset format).
    Classes:
        C1: Normal
        C2: Osteopenia
        C3: Osteoporosis
    """
    def __init__(self, root_dir, split='train', img_size=(224, 224)):
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g. data/train_ds/train_ds)
            split (str): 'train' or 'test'
            img_size (tuple): Target image size for resizing.
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
        # Load all png and jpg files
        self.image_paths = glob(os.path.join(root_dir, '*.png')) + glob(os.path.join(root_dir, '*.jpg'))
        if not self.image_paths:
            # Fallback if there are subdirectories
            self.image_paths = glob(os.path.join(root_dir, '**', '*.png'), recursive=True) + glob(os.path.join(root_dir, '**', '*.jpg'), recursive=True)
            
        # Map C1, C2, C3 to 0, 1, 2
        self.class_map = {'C1': 0, 'C2': 1, 'C3': 2}
        
        # Determine transformations based on split
        # To avoid aspect ratio distortion (200x100 to 224x224), we pad it to square (200x200) first
        pad_to_square = transforms.Pad((0, 50, 0, 50), fill=0) # left, top, right, bottom
        
        if split == 'train':
            self.transform = transforms.Compose([
                pad_to_square,
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                pad_to_square,
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Extract label from filename (e.g. C1_0_0.png)
        filename = os.path.basename(img_path)
        class_prefix = filename.split('_')[0]
        
        label = self.class_map.get(class_prefix, 0)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
