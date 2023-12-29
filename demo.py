#!/usr/bin/env python3
"""
This script creates distance maps from the mean shapes in the data 
directory.  It should be executed in a virtual environment with all the 
dependencies installed.  To increase resolution, use the optional 
parameters, e.g.  --num_rays_x=224 and --num_rays_y=224
"""

import os

def call_cd_maps():
    """
    Prints and executes the distance maps creation.
    """
    classes = ["control", "coronal", "metopic", "sagittal"]
    for cls in classes:
        src_file = "cdmap/create_distance_map.py"
        input_ply = "data/mean_shapes/ms_" + cls + ".ply"
        output_file = "results_img_" + cls + ".png"
        system_call = f"python {src_file} {input_ply} --output_path_image={output_file}"
        print("Executing: " + system_call)
        os.system(system_call)

if __name__ == '__main__':
    print(__doc__)
    call_cd_maps()
