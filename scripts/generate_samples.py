import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import random

def generate_anatomical_dataset(count=100):
    os.makedirs('samples', exist_ok=True)
    for i in range(count):
        is_pos = np.random.random() > 0.4
        # Create brain background (Gray Oval)
        img = Image.new('L', (512, 512), 0)
        draw = ImageDraw.Draw(img)
        # Brain mask
        draw.ellipse([50, 50, 460, 460], fill=40)
        
        # Vessels (Complex structure)
        draw.line([256, 100, 256, 412], fill=160, width=5) # Baseline
        draw.line([100, 256, 412, 256], fill=160, width=5) # Circle
        
        if is_pos:
            # Add Aneurysm (Bright Blob)
            ay, ax = random.randint(150, 350), random.randint(150, 350)
            draw.ellipse([ax-15, ay-15, ax+15, ay+15], fill=255)
            name = f"pos_anatomical_{i}.png"
        else:
            name = f"neg_anatomical_{i}.png"
            
        # Add noise
        final_img = np.array(img) + np.random.normal(0, 5, (512, 512))
        final_img = np.clip(final_img, 0, 255).astype(np.uint8)
        
        Image.fromarray(final_img).save(f"samples/{name}")
        
    print(f"Generated {count} ANATOMICAL images in 'samples/' folder.")

generate_anatomical_dataset(100)
