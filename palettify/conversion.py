import numpy as np
from numba import njit, vectorize
from palettify.types import *
import numpy.typing as npt
from scipy import stats

@njit
def interpolate(color: np.ndarray, closest1: np.ndarray, closest2: np.ndarray) -> np.ndarray:
    d:np.float32 = np.sum(np.square(closest2 - closest1), dtype=np.float32)
    
    if d == 0:
        return closest1
    
    f = np.dot(color - closest1, closest2-closest1) / d
    if f <= 0:
        return closest1
    elif f >= 1:
        return closest2

    return closest1 + f * (closest2 - closest1)
    
@njit
def findClosestColors(rgb: np.ndarray, palette: np.ndarray, exp:np.float32=15) -> np.ndarray:
    """Find the closest color in the palette to the given RGB color."""
    # Calculate squared distances for all palette colors
    # dists:npt.NDArray = np.sqrt(np.sum(np.square(palette - rgb), axis=1))
    dists:npt.NDArray = np.sum(np.square(palette - rgb), axis=1)
    # Find the index of the minimum distance
    
    fac = np.interp(dists, (dists.min(), dists.max()), (1, 0)) ** exp
    
    # fac = stats.norm.pdf(dists, dists.min(), 15)
    
    # print(fac)
    # print(fac[:, np.newaxis])
    interp_colors = palette * fac[:, np.newaxis]
    # print(interp_colors)
    res = np.sum(interp_colors, axis=0) / np.sum(fac)
    # print(res)
    return res

@njit
def applyPalette(arr: ImageArray, palette: PaletteArray, exp:np.float32=15) -> ImageArray:
    """Map the image array to the closest color in the palette."""
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            # old = arr[i, j]
            arr[i, j] = findClosestColors(arr[i, j], palette, exp)
            
            # error = old - new
            
            # # Floyd steinberg dithering
            # if i + 1 < arr.shape[0]:
            #     arr[i+1, j] = arr[i+1, j] + error * 7 / 16
            # if i - 1 >= 0 and j + 1 < arr.shape[1]:
            #     arr[i-1, j+1] = arr[i-1, j+1] + error * 3 / 16
            # if j + 1 < arr.shape[1]:
            #     arr[i, j+1] = arr[i, j+1] + error * 5 / 16
            # if i + 1 < arr.shape[0] and j + 1 < arr.shape[1]:
            #     arr[i+1, j-1] = arr[i+1, j-1] + error * 1 / 16
                
            
    return arr


    # Load the image and apply the palette
