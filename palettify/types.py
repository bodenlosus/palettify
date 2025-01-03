from numba import types

RGBArray = types.Array(types.uint8, 1, 'C')  # 1D array for a single RGB color
PaletteArray = types.Array(types.uint8, 2, 'C')  # 2D array for the palette
ImageArray = types.Array(types.float32, 3, 'C')  # 3D array for image data (H x W x RGB)