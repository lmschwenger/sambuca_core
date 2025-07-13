"""Inversion handler for Sambuca using lookup tables and optimization methods."""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from ..lookup_table import LookUpTable


@dataclass
class InversionResult:
    """Results from an inversion operation.
    
    Attributes:
        parameters: Dictionary of inverted parameter values for each pixel.
        errors: Array of error values for each pixel.
        modeled_spectra: Array of modeled spectra for each pixel.
        timing: Dictionary with timing information.
        metadata: Additional metadata about the inversion.
    """
    parameters: Dict[str, NDArray[np.float64]]
    errors: NDArray[np.float64]
    modeled_spectra: NDArray[np.float64]
    timing: Dict[str, float]
    metadata: Dict[str, Any]


class InversionHandler:
    """Handler for performing inversions using lookup tables and optimization methods."""
    
    def __init__(self):
        """Initialize the inversion handler."""
        pass
    
    def invert_image_from_lookup_table(
        self,
        lookup_table: LookUpTable,
        observed_image: NDArray[np.float64],
        metric: str = 'euclidean',
        use_kdtree: bool = True,
        mask: Optional[NDArray[np.bool_]] = None,
        chunk_size: Optional[int] = None
    ) -> InversionResult:
        """Invert an image using a lookup table.
        
        Args:
            lookup_table: Pre-built LookUpTable object.
            observed_image: Observed reflectance image with shape (height, width, bands).
            metric: Distance metric ('euclidean', 'rmse', 'sam').
            use_kdtree: Whether to use KD-tree for faster lookups.
            mask: Optional boolean mask for valid pixels (True = valid).
            chunk_size: Process pixels in chunks of this size for memory efficiency.
            
        Returns:
            InversionResult containing inverted parameters and metadata.
            
        Raises:
            ValueError: If lookup table not built or image dimensions don't match.
        """
        start_time = time.time()
        
        # Validate inputs
        if not lookup_table.table_built:
            raise ValueError("Lookup table must be built before inversion")
            
        if observed_image.shape[2] != len(lookup_table.wavelengths):
            raise ValueError(f"Image has {observed_image.shape[2]} bands but lookup table expects {len(lookup_table.wavelengths)}")
        
        height, width, n_bands = observed_image.shape
        n_pixels = height * width
        
        # Create or validate mask
        if mask is None:
            mask = np.ones((height, width), dtype=bool)
        elif mask.shape != (height, width):
            raise ValueError(f"Mask shape {mask.shape} doesn't match image shape {(height, width)}")
        
        # Get valid pixel indices
        valid_indices = np.where(mask.ravel())[0]
        n_valid_pixels = len(valid_indices)
        
        if n_valid_pixels == 0:
            raise ValueError("No valid pixels found in mask")
        
        # Reshape image to (n_pixels, n_bands) and extract valid pixels
        image_reshaped = observed_image.reshape(n_pixels, n_bands)
        valid_spectra = image_reshaped[valid_indices]
        
        # Initialize output arrays
        param_results = {name: np.full((height, width), np.nan) for name in lookup_table.param_names}
        error_results = np.full((height, width), np.nan)
        modeled_results = np.full((height, width, n_bands), np.nan)
        
        # Build KD-tree if requested and metric supports it
        kdtree = None
        if use_kdtree and metric in ['euclidean', 'rmse']:
            print("Building KD-tree for fast lookups...")
            kdtree = cKDTree(lookup_table.spectra_array)
        
        # Process pixels (with optional chunking for memory efficiency)
        if chunk_size is None:
            chunk_size = min(10000, n_valid_pixels)  # Default chunk size
        
        n_chunks = (n_valid_pixels + chunk_size - 1) // chunk_size
        
        print(f"Processing {n_valid_pixels} valid pixels in {n_chunks} chunks...")
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_valid_pixels)
            
            chunk_indices = valid_indices[start_idx:end_idx]
            chunk_spectra = valid_spectra[start_idx:end_idx]
            
            # Invert pixels in this chunk
            chunk_params, chunk_errors, chunk_modeled = self._invert_pixels(
                chunk_spectra, lookup_table, metric, kdtree
            )
            
            # Store results
            for i, pixel_idx in enumerate(chunk_indices):
                row, col = divmod(pixel_idx, width)
                
                for param_name in lookup_table.param_names:
                    param_results[param_name][row, col] = chunk_params[param_name][i]
                
                error_results[row, col] = chunk_errors[i]
                modeled_results[row, col, :] = chunk_modeled[i]
        
        total_time = time.time() - start_time
        
        # Create result
        return InversionResult(
            parameters=param_results,
            errors=error_results,
            modeled_spectra=modeled_results,
            timing={
                'total': total_time,
                'per_pixel': total_time / n_valid_pixels if n_valid_pixels > 0 else 0
            },
            metadata={
                'n_valid_pixels': n_valid_pixels,
                'n_total_pixels': n_pixels,
                'image_shape': (height, width, n_bands),
                'metric': metric,
                'use_kdtree': use_kdtree and kdtree is not None,
                'chunk_size': chunk_size,
                'lut_size': len(lookup_table.param_array)
            }
        )
    
    def _invert_pixels(
        self,
        pixel_spectra: NDArray[np.float64],
        lookup_table: LookUpTable,
        metric: str,
        kdtree: Optional[cKDTree]
    ) -> tuple[Dict[str, NDArray[np.float64]], NDArray[np.float64], NDArray[np.float64]]:
        """Invert a batch of pixels using the lookup table.
        
        Args:
            pixel_spectra: Array of pixel spectra with shape (n_pixels, n_bands).
            lookup_table: The lookup table to use.
            metric: Distance metric to use.
            kdtree: Optional KD-tree for fast lookups.
            
        Returns:
            Tuple of (parameters_dict, errors, modeled_spectra).
        """
        n_pixels = pixel_spectra.shape[0]
        n_bands = pixel_spectra.shape[1]
        
        # Initialize output arrays
        param_results = {name: np.zeros(n_pixels) for name in lookup_table.param_names}
        error_results = np.zeros(n_pixels)
        modeled_results = np.zeros((n_pixels, n_bands))
        
        if kdtree is not None and metric in ['euclidean', 'rmse']:
            # Use KD-tree for fast lookups
            distances, indices = kdtree.query(pixel_spectra)
            
            for i in range(n_pixels):
                best_idx = indices[i]
                best_params = lookup_table.param_array[best_idx]
                best_spectrum = lookup_table.spectra_array[best_idx]
                
                # Store results
                for j, param_name in enumerate(lookup_table.param_names):
                    param_results[param_name][i] = best_params[j]
                
                if metric == 'rmse':
                    error_results[i] = np.sqrt(np.mean((best_spectrum - pixel_spectra[i]) ** 2))
                else:  # euclidean
                    error_results[i] = distances[i]
                
                modeled_results[i] = best_spectrum
        
        else:
            # Brute force search for all metrics
            for i in range(n_pixels):
                best_error = np.inf
                best_idx = 0
                
                pixel_spectrum = pixel_spectra[i]
                
                for lut_idx, lut_spectrum in enumerate(lookup_table.spectra_array):
                    if metric == 'rmse':
                        error = np.sqrt(np.mean((lut_spectrum - pixel_spectrum) ** 2))
                    elif metric == 'euclidean':
                        error = np.sqrt(np.sum((lut_spectrum - pixel_spectrum) ** 2))
                    elif metric == 'sam':
                        # Spectral Angle Mapper
                        dot_product = np.sum(lut_spectrum * pixel_spectrum)
                        norm_product = np.sqrt(np.sum(lut_spectrum ** 2) * np.sum(pixel_spectrum ** 2))
                        if norm_product < 1e-10:
                            error = np.pi / 2
                        else:
                            error = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
                    else:
                        raise ValueError(f"Unknown metric: {metric}")
                    
                    if error < best_error:
                        best_error = error
                        best_idx = lut_idx
                
                # Store best results
                best_params = lookup_table.param_array[best_idx]
                best_spectrum = lookup_table.spectra_array[best_idx]
                
                for j, param_name in enumerate(lookup_table.param_names):
                    param_results[param_name][i] = best_params[j]
                
                error_results[i] = best_error
                modeled_results[i] = best_spectrum
        
        return param_results, error_results, modeled_results