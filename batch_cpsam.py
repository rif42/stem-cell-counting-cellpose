import os
import glob
import time
import numpy as np
from cellpose import models, io

def main():
    # Define paths
    base_dir = r"e:\work\cellpose0"
    # User requested specific directory: processed-dataset/sub50/unzoomed/train
    target_dir = os.path.join(base_dir, "processed-dataset", "sub50", "unzoomed", "train")
    input_dir = target_dir
    output_dir = target_dir

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Model parameters
    # "run CPSAM" maps to the cpsam model
    # diameter:50, flow threshold:2, cellprob threshold:-0.5
    model_name = 'cpsam'
    diameter = 50
    flow_threshold = 2
    cellprob_threshold = -0.5

    print(f"Loading model: {model_name}")
    # Initialize model
    model = models.CellposeModel(gpu=True, pretrained_model=model_name)

    # Get list of images recursively
    # Assuming common image formats
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
    image_files = []
    for ext in extensions:
        # Non-recursive glob since we are targeting a specific leaf directory
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    # Sort files for consistent processing order
    image_files.sort()

    print(f"Found {len(image_files)} images in {input_dir}")

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        
        # Save explicitly to the same directory
        output_subdir = output_dir
            
        print(f"Processing {img_name}...")
        
        try:
            # Read image
            img = io.imread(img_path)
            
            # Run evaluation
            masks, flows, styles = model.eval(
                img, 
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold
                # channels=[0,0] removed as deprecated
            )

            # Save results as *_seg.npy
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(output_subdir, base_name)
            
            io.masks_flows_to_seg(img, masks, flows, output_path)
            print(f"Saved result to {output_path}_seg.npy")

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

if __name__ == "__main__":
    main()
