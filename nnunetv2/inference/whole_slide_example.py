#!/usr/bin/env python3
"""
Example script for using the predict_from_whole_slide function
"""

import torch
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def main():
    # Initialize the predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    # Initialize from a trained model folder
    # Replace with your actual model path
    model_folder = "/path/to/your/trained/model"
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),  # Use fold 0, or specify multiple folds
        checkpoint_name='checkpoint_final.pth'
    )
    
    # Path to your whole slide image
    slide_path = "/path/to/your/slide.tif"  # or .svs
    
    # Run prediction
    result = predictor.predict_from_whole_slide(
        slide_path=slide_path,
        tile_size=(1024, 1024),  # Size of tiles to extract
        step_size=256,            # Step size between tiles
        batch_size=32,            # Batch size for inference
        num_workers=4,            # Number of workers for data loading
        return_probabilities=True, # Set to True if you want probability maps
        detect_centroids=True,    # Enable centroid detection
        centroid_threshold=0.1    # Threshold for centroid detection
    )
    
    # Access results
    segmentation = result['segmentation']  # Binary segmentation mask
    probabilities = result.get('probabilities')  # Probability maps (if return_probabilities=True)
    centroids = result.get('centroids')  # List of detected centroids
    slide_properties = result['slide_properties']  # Slide metadata
    
    print(f"Segmentation shape: {segmentation.shape}")
    if probabilities is not None:
        print(f"Probability maps shape: {probabilities.shape}")
    print(f"Number of detected centroids: {len(centroids) if centroids else 0}")
    print(f"Slide properties: {slide_properties}")
    
    # Example: Get probability for class 1 (if it exists)
    if probabilities is not None and probabilities.shape[0] > 1:
        class1_prob = probabilities[1]  # Probability map for class 1
        print(f"Class 1 probability map shape: {class1_prob.shape}")
        print(f"Max probability for class 1: {np.max(class1_prob)}")
    
    # Example: Save results
    import cv2
    
    # Save segmentation mask
    cv2.imwrite('segmentation.png', segmentation * 255)
    
    # Save probability maps (normalize to 0-255)
    if probabilities is not None:
        for i in range(probabilities.shape[0]):
            prob_map = probabilities[i]
            prob_map_normalized = ((prob_map - prob_map.min()) / (prob_map.max() - prob_map.min()) * 255).astype(np.uint8)
            cv2.imwrite(f'probability_class_{i}.png', prob_map_normalized)
    
    # Save centroids as a text file
    if centroids:
        np.savetxt('centroids.txt', centroids, fmt='%d', header='x y')

if __name__ == "__main__":
    main() 