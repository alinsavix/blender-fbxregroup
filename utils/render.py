import bpy
import math
from typing import Optional, Tuple


def set_output_properties(scene: bpy.types.Scene,
                          resolution_percentage: int = 100,
                          output_file_path: str = "") -> None:
    scene.render.resolution_percentage = resolution_percentage

    if output_file_path:
        scene.render.filepath = output_file_path


def set_cycles_renderer(scene: bpy.types.Scene,
                        camera_object: bpy.types.Object,
                        num_samples: int,
                        use_denoising: bool = True,
                        use_motion_blur: bool = False,
                        use_transparent_bg: bool = False) -> None:
    scene.camera = camera_object

    scene.render.image_settings.file_format = 'PNG'
    scene.render.engine = 'CYCLES'
    scene.render.use_motion_blur = use_motion_blur

    scene.render.film_transparent = use_transparent_bg
    scene.view_layers[0].cycles.use_denoising = use_denoising

    scene.cycles.samples = num_samples
