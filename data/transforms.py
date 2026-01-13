import random
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Dict, Any, Optional, Union
import math


class Augmentation:
    """Base augmentation class"""
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray, bboxes: List[List[float]], labels: List[int]) -> Dict[str, Any]:
        if random.random() < self.p:
            return self.apply(image, bboxes, labels)
        return {'image': image, 'bboxes': bboxes, 'labels': labels}
    
    def apply(self, image: np.ndarray, bboxes: List[List[float]], labels: List[int]) -> Dict[str, Any]:
        raise NotImplementedError


class RandomPerspective(Augmentation):
    """Random perspective transformation"""
    def __init__(self, degrees: float = 0.0, translate: float = 0.1, scale: float = 0.5, 
                 shear: float = 0.0, perspective: float = 0.0, p: float = 0.5):
        super().__init__(p)
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
    
    def apply(self, image: np.ndarray, bboxes: List[List[float]], labels: List[int]) -> Dict[str, Any]:
        height, width = image.shape[:2]
        
        # Center
        C = np.eye(3)
        C[0, 2] = -width / 2
        C[1, 2] = -height / 2
        
        # Rotation
        R = np.eye(3)
        angle = random.uniform(-self.degrees, self.degrees)
        scale = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)
        
        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        
        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height
        
        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)
        
        # Combined transformation matrix
        M = T @ S @ R @ P @ C
        M = M[:2]
        
        # Apply transformation to image
        warped = cv2.warpAffine(image, M, dsize=(width, height), 
                               borderValue=(114, 114, 114))
        
        # Apply transformation to bounding boxes
        transformed_bboxes = []
        transformed_labels = []
        
        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox
            
            # Convert to corners
            corners = np.array([
                [x1, y1, 1],
                [x2, y1, 1],
                [x1, y2, 1],
                [x2, y2, 1]
            ]).T
            
            # Transform corners
            transformed_corners = M @ corners
            
            # Get new bounding box
            x_min = transformed_corners[0].min()
            x_max = transformed_corners[0].max()
            y_min = transformed_corners[1].min()
            y_max = transformed_corners[1].max()
            
            # Check if box is valid
            if x_max - x_min > 1 and y_max - y_min > 1:
                transformed_bboxes.append([x_min, y_min, x_max, y_max])
                transformed_labels.append(label)
        
        return {'image': warped, 'bboxes': transformed_bboxes, 'labels': transformed_labels}


class Mosaic(Augmentation):
    """Mosaic augmentation - combine 4 images"""
    def __init__(self, p: float = 0.5, img_size: Tuple[int, int] = (640, 640)):
        super().__init__(p)
        self.img_size = img_size
    
    def apply(self, image: np.ndarray, bboxes: List[List[float]], labels: List[int], 
              images: List[np.ndarray], bboxes_list: List[List[List[float]]], 
              labels_list: List[List[int]]) -> Dict[str, Any]:
        """
        Apply mosaic augmentation
        
        Args:
            images: List of 4 images
            bboxes_list: List of 4 bbox lists
            labels_list: List of 4 label lists
        """
        if len(images) < 4:
            return {'image': image, 'bboxes': bboxes, 'labels': labels}
        
        # Randomly select 3 additional images
        indices = random.sample(range(len(images)), 3)
        selected_images = [image] + [images[i] for i in indices]
        selected_bboxes = [bboxes] + [bboxes_list[i] for i in indices]
        selected_labels = [labels] + [labels_list[i] for i in indices]
        
        # Create output image
        output_img = np.full((self.img_size[0] * 2, self.img_size[1] * 2, 3), 114, dtype=np.uint8)
        
        # Random center
        cx = int(random.uniform(self.img_size[0] * 0.5, self.img_size[0] * 1.5))
        cy = int(random.uniform(self.img_size[1] * 0.5, self.img_size[1] * 1.5))
        
        output_bboxes = []
        output_labels = []
        
        # Place each image in a quadrant
        for i, (img, img_bboxes, img_labels) in enumerate(zip(selected_images, selected_bboxes, selected_labels)):
            h, w = img.shape[:2]
            
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(cx - w, 0), max(cy - h, 0), cx, cy
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = cx, max(cy - h, 0), min(cx + w, self.img_size[1] * 2), cy
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(cx - w, 0), cy, cx, min(self.img_size[0] * 2, cy + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
            else:  # bottom-right
                x1a, y1a, x2a, y2a = cx, cy, min(cx + w, self.img_size[1] * 2), min(self.img_size[0] * 2, cy + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)
            
            # Place image
            output_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # Adjust bounding boxes
            padw = x1a - x1b
            padh = y1a - y1b
            
            for bbox, label in zip(img_bboxes, img_labels):
                x1, y1, x2, y2 = bbox
                
                # Adjust coordinates
                x1 = max(x1b, min(x1, x2b)) + padw
                y1 = max(y1b, min(y1, y2b)) + padh
                x2 = max(x1b, min(x2, x2b)) + padw
                y2 = max(y1b, min(y2, y2b)) + padh
                
                # Check if box is valid
                if x2 - x1 > 1 and y2 - y1 > 1:
                    output_bboxes.append([x1, y1, x2, y2])
                    output_labels.append(label)
        
        # Resize to target size
        output_img = cv2.resize(output_img, self.img_size)
        
        # Adjust bounding boxes for resize
        scale_x = self.img_size[1] / (self.img_size[1] * 2)
        scale_y = self.img_size[0] / (self.img_size[0] * 2)
        
        final_bboxes = []
        for bbox in output_bboxes:
            x1, y1, x2, y2 = bbox
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y
            final_bboxes.append([x1, y1, x2, y2])
        
        return {'image': output_img, 'bboxes': final_bboxes, 'labels': output_labels}


class MixUp(Augmentation):
    """MixUp augmentation - blend two images"""
    def __init__(self, p: float = 0.5, alpha: float = 0.8):
        super().__init__(p)
        self.alpha = alpha
    
    def apply(self, image1: np.ndarray, bboxes1: List[List[float]], labels1: List[int],
              image2: np.ndarray, bboxes2: List[List[float]], labels2: List[int]) -> Dict[str, Any]:
        """
        Apply MixUp augmentation
        
        Args:
            image1, bboxes1, labels1: First image and annotations
            image2, bboxes2, labels2: Second image and annotations
        """
        # Random lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix images
        mixed_image = (image1 * lam + image2 * (1 - lam)).astype(np.uint8)
        
        # Combine bounding boxes and labels
        mixed_bboxes = bboxes1 + bboxes2
        mixed_labels = labels1 + labels2
        
        # Adjust labels for mixup (optional: could add mixup label encoding)
        
        return {'image': mixed_image, 'bboxes': mixed_bboxes, 'labels': mixed_labels}


class AlbumentationsWrapper:
    """Wrapper for Albumentations augmentations"""
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms
    
    def __call__(self, image: np.ndarray, bboxes: List[List[float]], labels: List[int]) -> Dict[str, Any]:
        # Convert bboxes to albumentations format
        albu_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # Convert to [x_min, y_min, x_max, y_max] format
            albu_bboxes.append([x1, y1, x2, y2])
        
        # Apply transformations
        transformed = self.transforms(image=image, bboxes=albu_bboxes, labels=labels)
        
        return transformed


class TrainTransforms:
    """Training transformations for object detection"""
    def __init__(self, 
                 img_size: Tuple[int, int] = (640, 640),
                 hsv_h: float = 0.015,
                 hsv_s: float = 0.7,
                 hsv_v: float = 0.4,
                 degrees: float = 0.0,
                 translate: float = 0.1,
                 scale: float = 0.5,
                 shear: float = 0.0,
                 perspective: float = 0.0,
                 flipud: float = 0.0,
                 fliplr: float = 0.5,
                 mosaic: float = 1.0,
                 mixup: float = 0.0,
                 copy_paste: float = 0.0):
        """
        Initialize training transforms
        
        Args:
            img_size: Target image size (height, width)
            hsv_h: Hue augmentation factor
            hsv_s: Saturation augmentation factor
            hsv_v: Value augmentation factor
            degrees: Rotation degrees
            translate: Translation factor
            scale: Scale factor
            shear: Shear factor
            perspective: Perspective factor
            flipud: Vertical flip probability
            fliplr: Horizontal flip probability
            mosaic: Mosaic augmentation probability
            mixup: Mixup augmentation probability
            copy_paste: Copy-paste augmentation probability
        """
        self.img_size = img_size
        self.mosaic_prob = mosaic
        self.mixup_prob = mixup
        
        # Basic augmentations
        self.basic_augmentations = A.Compose([
            A.RandomResizedCrop(height=img_size[0], width=img_size[1], scale=(0.5, 1.0)),
            A.HorizontalFlip(p=fliplr),
            A.VerticalFlip(p=flipud),
            A.ShiftScaleRotate(shift_limit=translate, scale_limit=scale, 
                             rotate_limit=degrees, p=0.5),
            A.Perspective(scale=perspective, p=0.5),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Equalize(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.RandomGamma(),
                A.HueSaturationValue(hue_shift_limit=hsv_h, sat_shift_limit=hsv_s, 
                                   val_shift_limit=hsv_v),
            ], p=0.5),
            A.CoarseDropout(max_holes=8, max_height=img_size[0]//20, 
                          max_width=img_size[1]//20, p=0.5),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),  # Normalize to [0, 1]
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        
        # Mosaic augmentation
        self.mosaic_aug = Mosaic(p=1.0, img_size=img_size)
        
        # MixUp augmentation
        self.mixup_aug = MixUp(p=1.0)
        
        # Perspective transformation
        self.perspective_aug = RandomPerspective(
            degrees=degrees, translate=translate, scale=scale,
            shear=shear, perspective=perspective, p=0.5
        )
    
    def __call__(self, image: np.ndarray, bboxes: List[List[float]], labels: List[int],
                 dataset: Optional[Any] = None, idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Apply training transformations
        
        Args:
            image: Input image
            bboxes: Bounding boxes in normalized xyxy format
            labels: Class labels
            dataset: Optional dataset for mosaic/mixup
            idx: Optional index in dataset for mosaic/mixup
        
        Returns:
            Transformed image and annotations
        """
        # Apply HSV augmentations
        image = self._apply_hsv(image)
        
        # Apply mosaic if enabled
        if random.random() < self.mosaic_prob and dataset is not None and idx is not None:
            # Get 3 random indices for mosaic
            indices = [idx] + random.sample(range(len(dataset)), 3)
            images = []
            bboxes_list = []
            labels_list = []
            
            for i in indices:
                img, ann = dataset[i]
                img = img.permute(1, 2, 0).numpy() * 255
                img = img.astype(np.uint8)
                images.append(img)
                bboxes_list.append(ann['bboxes'].numpy())
                labels_list.append(ann['labels'].numpy())
            
            result = self.mosaic_aug.apply(
                images[0], bboxes_list[0], labels_list[0],
                images[1:], bboxes_list[1:], labels_list[1:]
            )
            image = result['image']
            bboxes = result['bboxes']
            labels = result['labels']
        
        # Apply perspective transformation
        result = self.perspective_aug(image, bboxes, labels)
        image = result['image']
        bboxes = result['bboxes']
        labels = result['labels']
        
        # Apply basic augmentations
        result = self.basic_augmentations(image=image, bboxes=bboxes, labels=labels)
        image = result['image']
        bboxes = result['bboxes']
        labels = result['labels']
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Convert bboxes to tensor
        if len(bboxes) > 0:
            bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        
        return image_tensor, {'boxes': bboxes_tensor, 'labels': labels_tensor}
    
    def _apply_hsv(self, image: np.ndarray) -> np.ndarray:
        """Apply HSV augmentations"""
        # Random HSV augmentation
        r = np.random.uniform(-1, 1, 3) * [0.015, 0.7, 0.4] + 1
        
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        
        # Apply transformations
        dtype = image.dtype
        x = np.arange(0, 256, dtype=np.int16)
        
        # Hue
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        hue = cv2.LUT(hue, lut_hue)
        
        # Saturation
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        sat = cv2.LUT(sat, lut_sat)
        
        # Value
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        val = cv2.LUT(val, lut_val)
        
        # Merge back
        image_hsv = cv2.merge((hue, sat, val)).astype(dtype)
        image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
        
        return image


class ValTransforms:
    """Validation transformations for object detection"""
    def __init__(self, img_size: Tuple[int, int] = (640, 640)):
        """
        Initialize validation transforms
        
        Args:
            img_size: Target image size (height, width)
        """
        self.img_size = img_size
        
        self.transforms = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),  # Normalize to [0, 1]
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    def __call__(self, image: np.ndarray, bboxes: List[List[float]], labels: List[int]) -> Dict[str, Any]:
        """
        Apply validation transformations
        
        Returns:
            Transformed image and annotations
        """
        # Apply transformations
        result = self.transforms(image=image, bboxes=bboxes, labels=labels)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(result['image']).permute(2, 0, 1).float()
        
        # Convert bboxes to tensor
        if len(result['bboxes']) > 0:
            bboxes_tensor = torch.tensor(result['bboxes'], dtype=torch.float32)
            labels_tensor = torch.tensor(result['labels'], dtype=torch.long)
        else:
            bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        
        return image_tensor, {'boxes': bboxes_tensor, 'labels': labels_tensor}


class TestTransforms:
    """Test/inference transformations for object detection"""
    def __init__(self, img_size: Tuple[int, int] = (640, 640), stride: int = 32):
        """
        Initialize test transforms
        
        Args:
            img_size: Target image size (height, width)
            stride: Model stride for padding
        """
        self.img_size = img_size
        self.stride = stride
        
        self.transforms = A.Compose([
            A.LongestMaxSize(max_size=max(img_size)),
            A.PadIfNeeded(
                min_height=img_size[0],
                min_width=img_size[1],
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114)
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),  # Normalize to [0, 1]
        ])
    
    def __call__(self, image: np.ndarray) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply test transformations
        
        Returns:
            Transformed image and metadata
        """
        # Get original size
        orig_h, orig_w = image.shape[:2]
        
        # Apply transformations
        result = self.transforms(image=image)
        transformed_img = result['image']
        
        # Get padding
        pad_h = (self.img_size[0] - transformed_img.shape[0]) % self.stride
        pad_w = (self.img_size[1] - transformed_img.shape[1]) % self.stride
        
        if pad_h > 0 or pad_w > 0:
            transformed_img = cv2.copyMakeBorder(
                transformed_img, 0, pad_h, 0, pad_w,
                cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )
        
        # Convert to tensor
        image_tensor = torch.from_numpy(transformed_img).permute(2, 0, 1).float()
        
        # Prepare metadata
        metadata = {
            'orig_size': (orig_h, orig_w),
            'padded_size': transformed_img.shape[:2],
            'scale': min(self.img_size[0] / orig_h, self.img_size[1] / orig_w),
            'pad': (0, pad_w, 0, pad_h)  # top, right, bottom, left
        }
        
        return image_tensor, metadata
