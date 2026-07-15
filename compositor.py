import cv2
import numpy as np
from PIL import Image

class VPPCompositor:
    """
    Handles image and video processing tasks:
    - Extracting stabilized cropped patches (ROIs) centered on tracked masks.
    - Overlaying the brand asset onto the first crop.
    - Alpha-blending edited patches back to full-resolution frames using a blurred mask.
    """
    def __init__(self, crop_width=480, crop_height=480, blur_kernel_size=25):
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.blur_kernel_size = blur_kernel_size
        
    def calculate_centroids(self, masks, total_frames):
        """
        Computes the center (centroid) of the mask for each frame.
        Interpolates/fills missing centroids to maintain stability.
        """
        centroids = {}
        last_known = None
        
        # Phase 1: Compute raw centroids
        for i in range(total_frames):
            mask = masks.get(i)
            if mask is not None and np.any(mask > 0):
                y_idx, x_idx = np.where(mask > 0)
                cy = int(np.mean(y_idx))
                cx = int(np.mean(x_idx))
                centroids[i] = (cx, cy)
                last_known = (cx, cy)
            else:
                centroids[i] = None
                
        # Phase 2: Interpolate/fill missing centroids
        # Forward fill
        for i in range(total_frames):
            if centroids[i] is None:
                centroids[i] = last_known
                
        # Backward fill
        first_known = None
        for i in range(total_frames):
            if centroids[i] is not None:
                first_known = centroids[i]
                break
                
        for i in range(total_frames):
            if centroids[i] is None:
                centroids[i] = first_known
                
        return centroids

    def extract_stabilized_crops(self, frames, centroids):
        """
        Crops stabilized patches from video frames centered around the centroids.
        Returns:
            list of crop_patches (numpy BGR arrays)
            list of crop_coords (tuples of min_x, min_y)
        """
        crop_patches = []
        crop_coords = []
        
        half_w = self.crop_width // 2
        half_h = self.crop_height // 2
        
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            cx, cy = centroids.get(i, (w // 2, h // 2))
            
            # Determine crop boundaries
            min_x = cx - half_w
            max_x = cx + half_w
            min_y = cy - half_h
            max_y = cy + half_h
            
            # Pad frame if boundaries exceed frame dimensions
            pad_left = max(0, -min_x)
            pad_right = max(0, max_x - w)
            pad_top = max(0, -min_y)
            pad_bottom = max(0, max_y - h)
            
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                padded_frame = cv2.copyMakeBorder(
                    frame, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_REPLICATE
                )
                crop_patch = padded_frame[
                    min_y + pad_top:max_y + pad_top,
                    min_x + pad_left:max_x + pad_left
                ]
            else:
                crop_patch = frame[min_y:max_y, min_x:max_x]
                
            crop_patches.append(crop_patch)
            crop_coords.append((min_x, min_y))
            
        return crop_patches, crop_coords

    def crop_masks(self, masks, centroids, total_frames):
        """
        Crops the tracking masks in the exact same stabilized fashion as the frames.
        """
        crop_masks = []
        half_w = self.crop_width // 2
        half_h = self.crop_height // 2
        
        for i in range(total_frames):
            mask = masks.get(i)
            cx, cy = centroids.get(i)
            
            if mask is None:
                # If no mask existed, create empty mask patch
                crop_masks.append(np.zeros((self.crop_height, self.crop_width), dtype=np.uint8))
                continue
                
            h, w = mask.shape
            
            min_x = cx - half_w
            max_x = cx + half_w
            min_y = cy - half_h
            max_y = cy + half_h
            
            # Pad mask if boundaries exceed dimensions
            pad_left = max(0, -min_x)
            pad_right = max(0, max_x - w)
            pad_top = max(0, -min_y)
            pad_bottom = max(0, max_y - h)
            
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                padded_mask = cv2.copyMakeBorder(
                    mask, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=0
                )
                crop_mask = padded_mask[
                    min_y + pad_top:max_y + pad_top,
                    min_x + pad_left:max_x + pad_left
                ]
            else:
                crop_mask = mask[min_y:max_y, min_x:max_x]
                
            crop_masks.append(crop_mask)
            
        return crop_masks

    def blend_brand_asset_onto_crop(self, crop_img, crop_mask, brand_img):
        """
        Blends the brand asset onto the stabilized crop image using the local tracking mask.
        Warping or resizing is done to fit the target mask bounds.
        """
        # Ensure brand_img is resized to fit the mask bounding box
        y_idx, x_idx = np.where(crop_mask > 0)
        if len(y_idx) == 0:
            # Fallback if mask is empty in the first frame crop
            return crop_img.copy()
            
        min_y, max_y = y_idx.min(), y_idx.max()
        min_x, max_x = x_idx.min(), x_idx.max()
        
        box_w = max_x - min_x
        box_h = max_y - min_y
        
        if box_w <= 0 or box_h <= 0:
            return crop_img.copy()
            
        # Resize brand image to fit mask bounding box
        resized_brand = cv2.resize(brand_img, (box_w, box_h))
        
        composite = crop_img.copy()
        
        # Build local mask segment
        local_mask = crop_mask[min_y:max_y, min_x:max_x, np.newaxis]
        roi = composite[min_y:max_y, min_x:max_x]
        
        # Alpha blending on target segment
        blended = (resized_brand * local_mask) + (roi * (1 - local_mask))
        composite[min_y:max_y, min_x:max_x] = blended.astype(np.uint8)
        
        return composite

    def paste_and_blend_frame(self, original_frame, generated_patch, original_mask, crop_coord):
        """
        Pastes the generated patch back into the original frame.
        Applies a Gaussian-blurred mask to blend boundaries seamlessly.
        """
        h_orig, w_orig = original_frame.shape[:2]
        min_x, min_y = crop_coord
        
        # Build a full-size frame initialized with the original frame's content
        composite_frame = original_frame.copy()
        
        # Calculate placing coordinates, handling borders
        h_patch, w_patch = generated_patch.shape[:2]
        
        # Calculate boundaries inside the full-size image
        start_x = max(0, min_x)
        start_y = max(0, min_y)
        end_x = min(w_orig, min_x + w_patch)
        end_y = min(h_orig, min_y + h_patch)
        
        # Calculate corresponding slice inside the patch
        patch_start_x = start_x - min_x
        patch_start_y = start_y - min_y
        patch_end_x = patch_start_x + (end_x - start_x)
        patch_end_y = patch_start_y + (end_y - start_y)
        
        # Place generated patch
        composite_frame[start_y:end_y, start_x:end_x] = generated_patch[
            patch_start_y:patch_end_y, patch_start_x:patch_end_x
        ]
        
        # Apply Gaussian Blur to the binary mask to blur the edges
        if np.any(original_mask > 0):
            blurred_mask = cv2.GaussianBlur(
                original_mask.astype(np.float32), 
                (self.blur_kernel_size, self.blur_kernel_size), 0
            )
            # Normalize to 0-1
            blurred_mask = np.clip(blurred_mask / blurred_mask.max(), 0.0, 1.0)
        else:
            blurred_mask = np.zeros((h_orig, w_orig), dtype=np.float32)
            
        # Replicate mask to 3 channels
        blurred_mask_3d = blurred_mask[:, :, np.newaxis]
        
        # Blend original frame and composite frame
        final_frame = original_frame * (1.0 - blurred_mask_3d) + composite_frame * blurred_mask_3d
        
        return final_frame.astype(np.uint8)
