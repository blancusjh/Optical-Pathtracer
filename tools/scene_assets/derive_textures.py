"""Recreate project surface derivatives; writes only missing assets, never scenes.
Recipes and limitations: docs/assets/recipe_provenance.json.
Run from repository root. Requires NumPy and Pillow.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def linear(a):
    return np.where(a <= .04045, a / 12.92, ((a + .055) / 1.055) ** 2.4)


def encode(a):
    return np.where(a <= .0031308, 12.92 * a, 1.055 * np.maximum(a, 0) ** (1 / 2.4) - .055)


def derive(source, target, operation, multiplier=None):
    if target.exists():
        print('exists:', target)
        return
    if not source.is_file():
        raise FileNotFoundError(source)
    with Image.open(source) as original:
        im = original.convert('RGB')
        out = Image.new('RGB', im.size)
        for y in range(0, im.height, 256):
            a = np.asarray(im.crop((0, y, im.width, min(y + 256, im.height))), dtype=np.float32)
            if operation == 'roughness':
                a[:, :, 1] = np.uint8(25.5 + .35 * a[:, :, 1])
            else:
                a = linear(a / 255)
                if operation == 'steel':
                    grey = .18 + .50 * (a @ np.array([.2126, .7152, .0722], dtype=np.float32))
                    a = np.repeat(grey[:, :, None], 3, axis=2)
                else:
                    a *= np.asarray(multiplier, dtype=np.float32)
                a = encode(a) * 255
            out.paste(Image.fromarray(np.uint8(np.clip(a, 0, 255))), (0, y))
        target.parent.mkdir(parents=True, exist_ok=True)
        out.save(target, **({'quality': 97} if target.suffix == '.jpg' else {}))
    print('created:', target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    base = args.root.resolve() / 'assets'
    jobs = [
        ('polyhaven/white_plaster_02/white_plaster_02_diff_8k.jpg', 'polyhaven/white_plaster_02/ochre.jpg', 'pigment', [.64,.56,.44]),
        ('polyhaven/oak_veneer_01/oak_veneer_01_diff_16k.jpg', 'polyhaven/oak_veneer_01/fumed.jpg', 'pigment', [.24,.18,.13]),
        ('polyhaven/quatrefoil_jacquard_fabric/quatrefoil_jacquard_fabric_diff_8k.jpg', 'polyhaven/quatrefoil_jacquard_fabric/oxblood_8k.jpg', 'pigment', [.32,.23,.21]),
        ('polyhaven/round_wooden_table_01/textures/round_wooden_table_01_diff_8k.jpg', 'polyhaven/round_wooden_table_01/textures/dark_walnut_8k.jpg', 'pigment', [.40,.34,.29]),
        ('sketchfab/armor_man_and_horse/textures/Armor_Mat_baseColor.jpeg', 'sketchfab/armor_man_and_horse/textures/steel_reflectance.jpg', 'steel', None),
        ('sketchfab/armor_man_and_horse/textures/Armor_Mat_metallicRoughness.png', 'sketchfab/armor_man_and_horse/textures/polished_steel_roughness.png', 'roughness', None),
        ('lunar/lroc_color_2k.jpg', 'lunar/lroc_albedo_018.png', 'pigment', [.18/.48217439031845044]*3),
    ]
    Image.MAX_IMAGE_PIXELS = None
    for src, dst, operation, multiplier in jobs:
        derive(base / src, base / dst, operation, multiplier)


if __name__ == '__main__':
    main()
