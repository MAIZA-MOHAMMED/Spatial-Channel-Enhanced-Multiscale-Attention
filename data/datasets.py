import os
import cv2
import torch
import numpy as np
import albumentations as A
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import json
import random


class BaseDetectionDataset(Dataset):
    """Base class for object detection datasets"""
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 transform = None,
                 target_size: Tuple[int, int] = (640, 640),
                 cache_images: bool = False,
                 augment: bool = False):
        """
        Initialize dataset
        
        Args:
            root_dir: Root directory of dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Transformations to apply
            target_size: Target image size (height, width)
            cache_images: Whether to cache images in memory
            augment: Whether to apply augmentations
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.cache_images = cache_images
        self.augment = augment
        
        # Initialize cache
        self.image_cache = {}
        self.annotations_cache = {}
        
        # Load dataset
        self.image_paths = []
        self.annotations = []
        self._load_dataset()
        
        print(f"Loaded {len(self.image_paths)} images from {self.split} split")
    
    def _load_dataset(self):
        """Load dataset - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _load_image(self, idx: int) -> np.ndarray:
        """Load image from cache or disk"""
        image_path = self.image_paths[idx]
        
        if self.cache_images and idx in self.image_cache:
            return self.image_cache[idx].copy()
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.cache_images:
            self.image_cache[idx] = image.copy()
        
        return image
    
    def _load_annotation(self, idx: int) -> Dict[str, Any]:
        """Load annotation - to be implemented by subclasses"""
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get item from dataset
        
        Returns:
            image: Tensor [C, H, W]
            target: Dictionary with keys:
                - 'boxes': Tensor [N, 4] in xyxy format
                - 'labels': Tensor [N] class indices
                - 'image_id': int
                - 'orig_size': original image size (height, width)
                - 'img_path': image path
        """
        # Load image and annotation
        image = self._load_image(idx)
        annotation = self._load_annotation(idx)
        
        # Get original size
        orig_h, orig_w = image.shape[:2]
        
        # Apply transformations
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                bboxes=annotation['bboxes'],
                labels=annotation['labels']
            )
            
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Prepare target
        if len(bboxes) > 0:
            boxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([orig_h, orig_w]),
            'img_path': str(self.image_paths[idx])
        }
        
        return image_tensor, target
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        raise NotImplementedError
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution"""
        class_counts = {}
        for annotation in self.annotations:
            for label in annotation['labels']:
                class_name = self.get_class_names()[label]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts


class ExDarkDataset(BaseDetectionDataset):
    """ExDark dataset for low-light object detection"""
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 transform = None,
                 target_size: Tuple[int, int] = (640, 640),
                 cache_images: bool = False,
                 augment: bool = False):
        """
        Initialize ExDark dataset
        
        Classes:
            0: Bicycle, 1: Boat, 2: Bottle, 3: Bus, 4: Car, 
            5: Cat, 6: Chair, 7: Cup, 8: Dog, 9: Motorbike,
            10: People, 11: Table
        """
        self.class_names = [
            'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car',
            'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike',
            'People', 'Table'
        ]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        super().__init__(root_dir, split, transform, target_size, cache_images, augment)
    
    def _load_dataset(self):
        """Load ExDark dataset"""
        # Define split files
        split_file = self.root_dir / f'{self.split}.txt'
        
        if not split_file.exists():
            # Create splits if they don't exist
            self._create_splits()
        
        # Read image paths from split file
        with open(split_file, 'r') as f:
            image_names = [line.strip() for line in f.readlines()]
        
        # Load all images and annotations
        for image_name in image_names:
            # Image path
            image_path = self.root_dir / 'images' / image_name
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # Annotation path
            anno_path = self.root_dir / 'annotations' / image_name.replace('.jpg', '.txt')
            
            self.image_paths.append(image_path)
            self.annotations.append({
                'path': anno_path,
                'image_name': image_name
            })
    
    def _create_splits(self):
        """Create train/val/test splits if they don't exist"""
        images_dir = self.root_dir / 'images'
        all_images = list(images_dir.glob('*.jpg'))
        
        # Shuffle and split
        random.seed(42)
        random.shuffle(all_images)
        
        n_total = len(all_images)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        train_images = all_images[:n_train]
        val_images = all_images[n_train:n_train + n_val]
        test_images = all_images[n_train + n_val:]
        
        # Save splits
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        for split_name, split_images in splits.items():
            split_file = self.root_dir / f'{split_name}.txt'
            with open(split_file, 'w') as f:
                for img_path in split_images:
                    f.write(f"{img_path.name}\n")
            
            print(f"Created {split_name} split with {len(split_images)} images")
    
    def _load_annotation(self, idx: int) -> Dict[str, Any]:
        """Load ExDark annotation"""
        anno_info = self.annotations[idx]
        anno_path = anno_info['path']
        
        if idx in self.annotations_cache:
            return self.annotations_cache[idx].copy()
        
        bboxes = []
        labels = []
        
        if anno_path.exists():
            with open(anno_path, 'r') as f:
                for line in f.readlines():
                    # Format: class x_center y_center width height
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert to xyxy format
                        x1 = (x_center - width / 2)
                        y1 = (y_center - height / 2)
                        x2 = (x_center + width / 2)
                        y2 = (y_center + height / 2)
                        
                        bboxes.append([x1, y1, x2, y2])
                        labels.append(class_id)
        
        annotation = {
            'bboxes': bboxes,
            'labels': labels
        }
        
        if self.cache_images:
            self.annotations_cache[idx] = annotation.copy()
        
        return annotation
    
    def get_class_names(self) -> List[str]:
        return self.class_names


class VisDroneDataset(BaseDetectionDataset):
    """VisDrone2019 dataset for aerial object detection"""
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 transform = None,
                 target_size: Tuple[int, int] = (640, 640),
                 cache_images: bool = False,
                 augment: bool = False):
        """
        Initialize VisDrone dataset
        
        Classes:
            0: pedestrian, 1: people, 2: bicycle, 3: car,
            4: van, 5: truck, 6: tricycle, 7: awning-tricycle,
            8: bus, 9: motor
        """
        self.class_names = [
            'pedestrian', 'people', 'bicycle', 'car',
            'van', 'truck', 'tricycle', 'awning-tricycle',
            'bus', 'motor'
        ]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        super().__init__(root_dir, split, transform, target_size, cache_images, augment)
    
    def _load_dataset(self):
        """Load VisDrone dataset"""
        # Define split directory
        split_dir = self.root_dir / 'VisDrone2019-DET' / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Load images and annotations
        images_dir = split_dir / 'images'
        annotations_dir = split_dir / 'annotations'
        
        for image_path in sorted(images_dir.glob('*.jpg')):
            anno_path = annotations_dir / image_path.name.replace('.jpg', '.txt')
            
            self.image_paths.append(image_path)
            self.annotations.append({
                'path': anno_path,
                'image_name': image_path.name
            })
    
    def _load_annotation(self, idx: int) -> Dict[str, Any]:
        """Load VisDrone annotation"""
        anno_info = self.annotations[idx]
        anno_path = anno_info['path']
        
        if idx in self.annotations_cache:
            return self.annotations_cache[idx].copy()
        
        bboxes = []
        labels = []
        
        if anno_path.exists():
            with open(anno_path, 'r') as f:
                for line in f.readlines():
                    # Format: bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion
                    parts = line.strip().split(',')
                    if len(parts) >= 8:
                        # Filter based on visibility
                        truncation = int(parts[6])
                        occlusion = int(parts[7])
                        
                        # Only keep objects with truncation <= 2 and occlusion <= 2
                        if truncation <= 2 and occlusion <= 2:
                            x1 = float(parts[0])
                            y1 = float(parts[1])
                            width = float(parts[2])
                            height = float(parts[3])
                            category = int(parts[5])
                            
                            # Map category to class index (0-9)
                            if 1 <= category <= 10:  # Ignore ignored regions (0)
                                class_idx = category - 1  # Map to 0-9
                                
                                x2 = x1 + width
                                y2 = y1 + height
                                
                                bboxes.append([x1, y1, x2, y2])
                                labels.append(class_idx)
        
        annotation = {
            'bboxes': bboxes,
            'labels': labels
        }
        
        if self.cache_images:
            self.annotations_cache[idx] = annotation.copy()
        
        return annotation
    
    def get_class_names(self) -> List[str]:
        return self.class_names


class FYPDataset(BaseDetectionDataset):
    """FYP dataset for complex scene object detection"""
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 transform = None,
                 target_size: Tuple[int, int] = (640, 640),
                 cache_images: bool = False,
                 augment: bool = False):
        """
        Initialize FYP dataset
        
        Classes: Custom classes for the FYP dataset
        """
        # Define your custom classes here
        self.class_names = [
            'person', 'car', 'bus', 'truck', 'motorcycle',
            'bicycle', 'traffic light', 'stop sign', 'dog', 'cat'
        ]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        super().__init__(root_dir, split, transform, target_size, cache_images, augment)
    
    def _load_dataset(self):
        """Load FYP dataset"""
        # FYP dataset can be in various formats
        # Assuming COCO format for this example
        anno_file = self.root_dir / 'annotations' / f'{self.split}.json'
        
        if anno_file.exists():
            # COCO format
            self._load_coco_format(anno_file)
        else:
            # YOLO format
            self._load_yolo_format()
    
    def _load_coco_format(self, anno_file: Path):
        """Load dataset in COCO format"""
        with open(anno_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create mapping from image id to annotations
        img_id_to_annos = {}
        for anno in coco_data['annotations']:
            img_id = anno['image_id']
            if img_id not in img_id_to_annos:
                img_id_to_annos[img_id] = []
            img_id_to_annos[img_id].append(anno)
        
        # Create mapping from image id to image info
        img_id_to_info = {img['id']: img for img in coco_data['images']}
        
        # Load images and annotations
        for img_id, img_info in img_id_to_info.items():
            image_path = self.root_dir / 'images' / self.split / img_info['file_name']
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            self.image_paths.append(image_path)
            self.annotations.append({
                'image_info': img_info,
                'annos': img_id_to_annos.get(img_id, [])
            })
    
    def _load_yolo_format(self):
        """Load dataset in YOLO format"""
        images_dir = self.root_dir / 'images' / self.split
        labels_dir = self.root_dir / 'labels' / self.split
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        for image_path in sorted(images_dir.glob('*.jpg')):
            label_path = labels_dir / image_path.name.replace('.jpg', '.txt')
            
            self.image_paths.append(image_path)
            self.annotations.append({
                'path': label_path,
                'image_name': image_path.name
            })
    
    def _load_annotation(self, idx: int) -> Dict[str, Any]:
        """Load FYP annotation"""
        anno_info = self.annotations[idx]
        
        if idx in self.annotations_cache:
            return self.annotations_cache[idx].copy()
        
        bboxes = []
        labels = []
        
        if 'annos' in anno_info:
            # COCO format
            for anno in anno_info['annos']:
                # COCO bbox: [x, y, width, height]
                x, y, w, h = anno['bbox']
                category_id = anno['category_id']
                
                # Map to class index
                class_idx = self._coco_category_to_idx(category_id)
                if class_idx is not None:
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    bboxes.append([x1, y1, x2, y2])
                    labels.append(class_idx)
        else:
            # YOLO format
            label_path = anno_info['path']
            if label_path.exists():
                img_info = self.annotations[idx]
                img_width = img_info.get('width', 1)
                img_height = img_info.get('height', 1)
                
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convert to absolute coordinates
                            x_center *= img_width
                            y_center *= img_height
                            width *= img_width
                            height *= img_height
                            
                            # Convert to xyxy format
                            x1 = x_center - width / 2
                            y1 = y_center - height / 2
                            x2 = x_center + width / 2
                            y2 = y_center + height / 2
                            
                            bboxes.append([x1, y1, x2, y2])
                            labels.append(class_id)
        
        annotation = {
            'bboxes': bboxes,
            'labels': labels
        }
        
        if self.cache_images:
            self.annotations_cache[idx] = annotation.copy()
        
        return annotation
    
    def _coco_category_to_idx(self, category_id: int) -> Optional[int]:
        """Map COCO category ID to class index"""
        # Define your mapping here
        # This is an example mapping
        mapping = {
            1: 0,   # person
            2: 1,   # bicycle
            3: 2,   # car
            5: 3,   # bus
            7: 4,   # truck
            4: 5,   # motorcycle
            # Add more mappings as needed
        }
        return mapping.get(category_id, None)
    
    def get_class_names(self) -> List[str]:
        return self.class_names


class COCODataset(BaseDetectionDataset):
    """COCO dataset wrapper"""
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 transform = None,
                 target_size: Tuple[int, int] = (640, 640),
                 cache_images: bool = False,
                 augment: bool = False):
        """
        Initialize COCO dataset
        """
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        super().__init__(root_dir, split, transform, target_size, cache_images, augment)
    
    def _load_dataset(self):
        """Load COCO dataset"""
        anno_file = self.root_dir / 'annotations' / f'instances_{self.split}.json'
        
        if not anno_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {anno_file}")
        
        with open(anno_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings
        self.img_id_to_info = {img['id']: img for img in self.coco_data['images']}
        self.img_id_to_annos = {}
        
        for anno in self.coco_data['annotations']:
            img_id = anno['image_id']
            if img_id not in self.img_id_to_annos:
                self.img_id_to_annos[img_id] = []
            self.img_id_to_annos[img_id].append(anno)
        
        # Load images
        for img_id, img_info in self.img_id_to_info.items():
            image_path = self.root_dir / self.split / img_info['file_name']
            
            if not image_path.exists():
                # Try different paths
                image_path = self.root_dir / 'images' / self.split / img_info['file_name']
                if not image_path.exists():
                    print(f"Warning: Image not found: {img_info['file_name']}")
                    continue
            
            self.image_paths.append(image_path)
            self.annotations.append({
                'image_id': img_id,
                'annos': self.img_id_to_annos.get(img_id, [])
            })
    
    def _load_annotation(self, idx: int) -> Dict[str, Any]:
        """Load COCO annotation"""
        if idx in self.annotations_cache:
            return self.annotations_cache[idx].copy()
        
        anno_info = self.annotations[idx]
        bboxes = []
        labels = []
        
        for anno in anno_info['annos']:
            # Skip crowd annotations
            if anno.get('iscrowd', 0) == 1:
                continue
            
            # COCO bbox: [x, y, width, height]
            x, y, w, h = anno['bbox']
            category_id = anno['category_id']
            
            # COCO categories start from 1
            class_idx = category_id - 1
            if 0 <= class_idx < len(self.class_names):
                x1, y1, x2, y2 = x, y, x + w, y + h
                bboxes.append([x1, y1, x2, y2])
                labels.append(class_idx)
        
        annotation = {
            'bboxes': bboxes,
            'labels': labels
        }
        
        if self.cache_images:
            self.annotations_cache[idx] = annotation.copy()
        
        return annotation
    
    def get_class_names(self) -> List[str]:
        return self.class_names


class VOCDataset(BaseDetectionDataset):
    """Pascal VOC dataset wrapper"""
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 transform = None,
                 target_size: Tuple[int, int] = (640, 640),
                 cache_images: bool = False,
                 augment: bool = False,
                 year: str = '2012'):
        """
        Initialize Pascal VOC dataset
        """
        self.year = year
        self.class_names = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        super().__init__(root_dir, split, transform, target_size, cache_images, augment)
    
    def _load_dataset(self):
        """Load VOC dataset"""
        # Load split file
        split_file = self.root_dir / 'ImageSets' / 'Main' / f'{self.split}.txt'
        
        if not split_file.exists():
            raise FileNotFoundError(f"VOC split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            image_ids = [line.strip().split()[0] for line in f.readlines()]
        
        # Load images and annotations
        for image_id in image_ids:
            image_path = self.root_dir / 'JPEGImages' / f'{image_id}.jpg'
            anno_path = self.root_dir / 'Annotations' / f'{image_id}.xml'
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            self.image_paths.append(image_path)
            self.annotations.append({
                'path': anno_path,
                'image_id': image_id
            })
    
    def _load_annotation(self, idx: int) -> Dict[str, Any]:
        """Load VOC annotation from XML"""
        anno_info = self.annotations[idx]
        anno_path = anno_info['path']
        
        if idx in self.annotations_cache:
            return self.annotations_cache[idx].copy()
        
        bboxes = []
        labels = []
        
        if anno_path.exists():
            tree = ET.parse(anno_path)
            root = tree.getroot()
            
            # Get image size
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            for obj in root.findall('object'):
                # Skip difficult objects
                difficult = obj.find('difficult')
                if difficult is not None and int(difficult.text) == 1:
                    continue
                
                # Get class name
                class_name = obj.find('name').text
                if class_name not in self.class_to_idx:
                    continue
                
                class_idx = self.class_to_idx[class_name]
                
                # Get bounding box
                bbox = obj.find('bndbox')
                x1 = float(bbox.find('xmin').text)
                y1 = float(bbox.find('ymin').text)
                x2 = float(bbox.find('xmax').text)
                y2 = float(bbox.find('ymax').text)
                
                # Normalize coordinates
                x1 /= img_width
                y1 /= img_height
                x2 /= img_width
                y2 /= img_height
                
                bboxes.append([x1, y1, x2, y2])
                labels.append(class_idx)
        
        annotation = {
            'bboxes': bboxes,
            'labels': labels
        }
        
        if self.cache_images:
            self.annotations_cache[idx] = annotation.copy()
        
        return annotation
    
    def get_class_names(self) -> List[str]:
        return self.class_names


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Custom collate function for detection datasets
    
    Args:
        batch: List of (image, target) tuples
    
    Returns:
        images: Tensor [B, C, H, W]
        targets: List of target dictionaries
    """
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    return images, targets


def create_dataloader(dataset: Dataset,
                     batch_size: int = 16,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     pin_memory: bool = True,
                     drop_last: bool = False) -> DataLoader:
    """
    Create dataloader for detection dataset
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop last incomplete batch
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
