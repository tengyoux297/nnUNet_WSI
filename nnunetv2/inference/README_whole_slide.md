# Whole Slide Image Prediction with nnU-Net

This module provides functionality to perform segmentation on whole slide images (WSI) using nnU-Net. It supports TIF and SVS formats and includes centroid detection capabilities.

## Features

- **Whole Slide Processing**: Process large slide images by tiling them into manageable patches
- **Optional Probability Maps**: Return probability maps when requested (consistent with other nnU-Net functions)
- **Centroid Detection**: Optional detection of object centroids from segmentation results
- **Memory Efficient**: Uses background prefetching and batch processing
- **Multi-class Support**: Handles multiple segmentation classes

## Requirements

```bash
pip install openslide-python scikit-image torch torchvision
```

## Usage

### Basic Usage

```python
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch

# Initialize predictor
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True,
    device=torch.device('cuda'),
    verbose=False,
    allow_tqdm=True
)

# Load trained model
predictor.initialize_from_trained_model_folder(
    model_folder="/path/to/model",
    use_folds=(0,),
    checkpoint_name='checkpoint_final.pth'
)

# Predict on whole slide
result = predictor.predict_from_whole_slide(
    slide_path="/path/to/slide.tif",
    tile_size=(1024, 1024),
    step_size=256,
    batch_size=32,
    num_workers=4,
    return_probabilities=True,  # Set to True if you want probability maps
    detect_centroids=True,
    centroid_threshold=0.1
)
```

### Return Values

The function returns a dictionary with the following keys:

- **`segmentation`**: Full slide segmentation mask
- **`probabilities`**: Probability maps for all classes (if `return_probabilities=True`)
- **`centroids`**: List of detected centroids (if `detect_centroids=True`)
- **`slide_properties`**: Slide metadata (dimensions, tile info, etc.)
- **`tile_predictions`**: List of individual tile predictions
- **`tile_positions`**: List of tile positions

### Parameters

- **`slide_path`**: Path to the whole slide image file (.tif or .svs)
- **`tile_size`**: Size of tiles to extract (width, height)
- **`step_size`**: Step size between tiles (stride)
- **`batch_size`**: Batch size for model inference
- **`num_workers`**: Number of workers for data loading
- **`return_probabilities`**: Whether to return probability maps (default: False)
- **`detect_centroids`**: Whether to detect centroids from segmentation
- **`centroid_threshold`**: Threshold for centroid detection

## Example Output

```python
# Access results
segmentation = result['segmentation']  # Binary segmentation mask
probabilities = result.get('probabilities')  # Probability maps (if return_probabilities=True)
centroids = result.get('centroids')  # List of detected centroids

print(f"Segmentation shape: {segmentation.shape}")
if probabilities is not None:
    print(f"Probability maps shape: {probabilities.shape}")
    # Get probability for specific class
    class1_prob = probabilities[1]  # Probability map for class 1
    print(f"Max probability for class 1: {np.max(class1_prob)}")
```

## Centroid Detection

The centroid detection uses morphological operations to identify connected components in the segmentation mask and extract their centroids. This is useful for counting objects or analyzing spatial distributions.

## Memory Considerations

- Adjust `batch_size` based on your GPU memory
- Use smaller `tile_size` for very large slides
- Increase `num_workers` for faster data loading (but monitor memory usage)

## Performance Tips

1. Use GPU acceleration when available
2. Adjust batch size based on available memory
3. Use appropriate tile and step sizes for your use case
4. Consider using multiple workers for data loading

## File Formats

Supported whole slide image formats:
- **TIF/TIFF**: Tagged Image File Format
- **SVS**: Aperio ScanScope format

## Dependencies

- **OpenSlide**: For reading whole slide images
- **scikit-image**: For centroid detection and morphological operations
- **PyTorch**: For deep learning inference
- **NumPy**: For numerical operations 