"""Look-up table generation for Sambuca forward model.

This module provides functionality to create, save, and load look-up tables
of pre-computed forward model spectra for different parameter combinations.
"""

import itertools
import os
import pickle
import time
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ..forward_model import forward_model
from ..siop_manager import SIOPManager


class LookUpTable:
    """Look-up table for pre-computed Sambuca forward model spectra.

    This class builds and stores a look-up table of pre-computed spectra for
    different parameter combinations. The table can be saved and loaded for
    later use with inversion algorithms.
    """

    def __init__(self, siop_manager: SIOPManager, wavelengths: List[float], 
                 parameter_ranges: Dict[str, Tuple[float, float]], 
                 fixed_parameters: Optional[Dict[str, float]] = None):
        """Initialize the look-up table with SIOP manager and parameter ranges.

        Args:
            siop_manager: SIOPManager instance for getting spectral data.
            wavelengths: List of wavelengths to use for forward model calculations.
            parameter_ranges: Dictionary mapping parameter names to (min, max) tuples for LUT dimensions.
            fixed_parameters: Dictionary of fixed parameter values for parameters not in ranges.
        """
        self.siop_manager = siop_manager
        self.wavelengths = wavelengths
        self.parameter_ranges = parameter_ranges
        self.fixed_parameters = fixed_parameters or {}
        self.param_names = list(parameter_ranges.keys())
        self.bounds = list(parameter_ranges.values())
        self.points = {}  # Dict mapping parameter tuples to spectra
        self.param_array = None  # Array form of parameters
        self.spectra_array = None  # Array form of spectra
        self.table_built = False
        self.grid_shape = None  # Shape of the parameter grid
        self.param_values = []  # List of parameter values for each dimension
        self.in_memory = True  # Whether the full LUT is kept in memory
        self.lut_file = None  # Path to LUT file if using disk-based mode
        
        # Get SIOPs for the specified wavelengths
        self.siops = siop_manager.get_siops_for_wavelengths(wavelengths)
        
        # Validate that all required parameters are specified
        self._validate_parameters()

    def build_table(
            self,
            grid_size: Union[int, List[int]] = 10,
            progress_bar: bool = True,
            memory_optimized: bool = False,
            batch_size: int = 1000,
    ) -> None:
        """Build the look-up table by running the forward model for parameter combinations.

        Args:
            grid_size: Number of points along each parameter dimension (can be int or list).
            progress_bar: Whether to show a progress bar.
            memory_optimized: If True, reduce memory usage by not keeping all spectra in a dict.
            batch_size: Number of parameter combinations to process in each batch.

        Raises:
            ValueError: If no parameters are specified for LUT generation.
        """
        start_time = time.time()

        # Check if we have parameters to vary
        if not self.bounds:
            raise ValueError("No parameters specified for LUT generation")

        # Create parameter grid
        if isinstance(grid_size, int):
            grid_size = [grid_size] * len(self.bounds)

        self.param_values = []
        for i, bound in enumerate(self.bounds):
            low, high = bound
            self.param_values.append(np.linspace(low, high, grid_size[i]))

        self.grid_shape = tuple(len(values) for values in self.param_values)

        # Create all parameter combinations
        param_combinations = list(itertools.product(*self.param_values))
        total_combinations = len(param_combinations)

        # Convert to arrays for faster operations
        self.param_array = np.array(param_combinations)
        first_spectral_length = len(self.wavelengths)
        self.spectra_array = np.zeros((total_combinations, first_spectral_length))

        # Process in batches to avoid memory issues
        n_batches = (total_combinations + batch_size - 1) // batch_size

        if progress_bar:
            batch_iterator = tqdm(range(n_batches), desc="Building LUT")
        else:
            batch_iterator = range(n_batches)

        # Process each batch
        for batch_idx in batch_iterator:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_combinations)

            for i in range(start_idx, end_idx):
                params = tuple(self.param_array[i])
                
                # Create parameter dictionary
                param_dict = {}
                for j, param_name in enumerate(self.param_names):
                    param_dict[param_name] = params[j]

                # Run forward model with current parameters
                fwd_params = self._get_forward_model_params(param_dict)
                results = forward_model(**fwd_params)

                # Store results
                if not memory_optimized:
                    self.points[params] = results.rrs

                self.spectra_array[i] = results.rrs

        # Keep memory optimization setting
        self.in_memory = not memory_optimized

        self.table_built = True
        elapsed_time = time.time() - start_time
        print(f"Look-up table built with {total_combinations} parameter combinations in {elapsed_time:.2f} seconds")

        # Print memory usage estimate
        param_mem = self.param_array.nbytes / (1024 * 1024)
        spectra_mem = self.spectra_array.nbytes / (1024 * 1024)
        total_mem = param_mem + spectra_mem

        if not memory_optimized:
            dict_mem = total_combinations * (first_spectral_length * 8 + 64) / (1024 * 1024)  # Rough estimate
            total_mem += dict_mem

        print(f"Estimated memory usage: {total_mem:.2f} MB")
        
    def _get_forward_model_params(self, param_dict: Dict[str, float]) -> Dict[str, Any]:
        """Get forward model parameters from LUT parameters.
        
        Args:
            param_dict: Dictionary of LUT parameters (variable parameters only).
            
        Returns:
            Dictionary of parameters for forward_model() function.
        """
        # Start with required SIOP data
        fwd_params = {
            'wavelengths': self.wavelengths,
            'num_bands': len(self.wavelengths),
            'a_water': self.siops['a_water'] if 'a_water' in self.siops else self.siops.get('water_absorption', None),
            'a_ph_star': self.siops['a_ph_star'] if 'a_ph_star' in self.siops else self.siops.get('phytoplankton_absorption', None),
            'substrate1': self.siops['substrate1'] if 'substrate1' in self.siops else self.siops.get('sand_substrate', None),
        }
        
        # Add fixed parameters
        fwd_params.update(self.fixed_parameters)
        
        # Add variable LUT parameters (will override fixed if there are conflicts, but validation prevents this)
        fwd_params.update(param_dict)
        
        # Add optional substrate2 if available
        if 'substrate2' in self.siops:
            fwd_params['substrate2'] = self.siops['substrate2']
        elif 'seagrass_substrate' in self.siops:
            fwd_params['substrate2'] = self.siops['seagrass_substrate']
            
        return fwd_params
    
    def _validate_parameters(self) -> None:
        """Validate that all required forward model parameters are specified.
        
        Raises:
            ValueError: If required parameters are missing.
        """
        required_params = {'chl', 'cdom', 'nap', 'depth', 'substrate_fraction'}
        
        # Get all specified parameters (both variable and fixed)
        all_params = set(self.parameter_ranges.keys()) | set(self.fixed_parameters.keys())
        
        missing_params = required_params - all_params
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}. "
                           f"Must be specified in either parameter_ranges or fixed_parameters.")
        
        # Check for parameter conflicts (same param in both ranges and fixed)
        conflicts = set(self.parameter_ranges.keys()) & set(self.fixed_parameters.keys())
        if conflicts:
            raise ValueError(f"Parameters cannot be both variable and fixed: {conflicts}")

    def save(self, filename: str, compressed: bool = True) -> None:
        """Save the look-up table to a file.

        Args:
            filename: Path to save the LUT.
            compressed: Whether to use compressed format (slower but smaller file).
        """
        self.lut_file = filename

        if compressed:
            # Use numpy's compressed format for arrays
            np.savez_compressed(
                filename + "_arrays",
                param_array=self.param_array,
                spectra_array=self.spectra_array,
                grid_shape=np.array(self.grid_shape),
            )

            # Save parameter values separately
            param_values_dict = {f"param_values_{i}": values for i, values in enumerate(self.param_values)}
            np.savez_compressed(filename + "_param_values", **param_values_dict)

            # Save other attributes with pickle
            attrs_to_save = {
                'wavelengths': self.wavelengths,
                'parameter_ranges': self.parameter_ranges,
                'fixed_parameters': self.fixed_parameters,
                'param_names': self.param_names,
                'bounds': self.bounds,
                'table_built': self.table_built,
                'in_memory': self.in_memory,
            }

            with open(filename + "_attrs", 'wb') as f:
                pickle.dump(attrs_to_save, f)

            print(f"Look-up table saved to {filename} (split into multiple files)")

        else:
            # Standard pickle approach (all in one file)
            data_to_save = {
                'wavelengths': self.wavelengths,
                'parameter_ranges': self.parameter_ranges,
                'fixed_parameters': self.fixed_parameters,
                'param_names': self.param_names,
                'bounds': self.bounds,
                'param_array': self.param_array,
                'spectra_array': self.spectra_array,
                'points': self.points if self.in_memory else None,
                'table_built': self.table_built,
                'grid_shape': self.grid_shape,
                'param_values': self.param_values,
                'in_memory': self.in_memory,
            }

            with open(filename, 'wb') as f:
                pickle.dump(data_to_save, f)

            print(f"Look-up table saved to {filename}")

    @classmethod
    def load(cls, filename: str, siop_manager: SIOPManager, in_memory: bool = None) -> 'LookUpTable':
        """Load a look-up table from a file.

        Args:
            filename: Path to the saved LUT file.
            siop_manager: SIOPManager instance for spectral data.
            in_memory: Override the saved setting for keeping points dictionary in memory.

        Returns:
            Loaded LookUpTable object.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        load_start = time.time()

        # Check if this is a split compressed format
        is_compressed = os.path.exists(filename + "_arrays.npz")

        if is_compressed:
            # Load arrays
            arrays_data = np.load(filename + "_arrays.npz")
            param_array = arrays_data['param_array']
            spectra_array = arrays_data['spectra_array']
            grid_shape = tuple(arrays_data['grid_shape'])

            # Load parameter values
            param_values_data = np.load(filename + "_param_values.npz")
            param_values = []
            i = 0
            while f"param_values_{i}" in param_values_data:
                param_values.append(param_values_data[f"param_values_{i}"])
                i += 1

            # Load other attributes
            with open(filename + "_attrs", 'rb') as f:
                attrs_data = pickle.load(f)

            # Create instance
            lut = cls(siop_manager, attrs_data['wavelengths'], attrs_data['parameter_ranges'], 
                     attrs_data.get('fixed_parameters', {}))
            lut.param_names = attrs_data['param_names']
            lut.bounds = attrs_data['bounds']
            lut.table_built = attrs_data['table_built']
            lut.in_memory = in_memory if in_memory is not None else attrs_data['in_memory']
            lut.param_array = param_array
            lut.spectra_array = spectra_array
            lut.grid_shape = grid_shape
            lut.param_values = param_values
            lut.lut_file = filename

            # Reconstruct points dictionary if needed
            if lut.in_memory:
                lut.points = {}
                for i, params in enumerate(param_array):
                    lut.points[tuple(params)] = spectra_array[i]
        else:
            # Standard pickle format
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Look-up table file {filename} not found")

            with open(filename, 'rb') as f:
                data = pickle.load(f)

            lut = cls(siop_manager, data['wavelengths'], data['parameter_ranges'], 
                     data.get('fixed_parameters', {}))
            lut.param_names = data['param_names']
            lut.bounds = data['bounds']
            lut.param_array = data['param_array']
            lut.spectra_array = data['spectra_array']
            lut.table_built = data['table_built']
            lut.grid_shape = data['grid_shape']
            lut.param_values = data['param_values']
            lut.in_memory = in_memory if in_memory is not None else data['in_memory']
            lut.lut_file = filename

            # Load points if available and needed
            if lut.in_memory:
                if data['points'] is not None:
                    lut.points = data['points']
                else:
                    # Reconstruct from arrays
                    lut.points = {}
                    for i, params in enumerate(lut.param_array):
                        lut.points[tuple(params)] = lut.spectra_array[i]


        load_time = time.time() - load_start
        print(f"Look-up table loaded in {load_time:.2f} seconds")
        print(f"LUT contains {len(lut.param_array)} parameter combinations")

        return lut


    def get_spectra_cube(self) -> Tuple[NDArray[np.float64], List[NDArray[np.float64]]]:
        """Get the pre-computed spectra as a parameter cube.

        This function organizes the pre-computed spectra into a multi-dimensional
        array (cube) corresponding to the parameter grid.

        Returns:
            Tuple containing:
                - Multi-dimensional array of spectra
                - List of parameter value arrays for each dimension

        Raises:
            ValueError: If the look-up table has not been built yet.
        """
        if not self.table_built:
            raise ValueError("Look-up table not built yet, call build_table() first")

        # Get number of wavelengths from any entry
        n_wavelengths = self.spectra_array.shape[1]

        # Create output array with shape (dim1, dim2, ..., n_wavelengths)
        cube_shape = list(self.grid_shape) + [n_wavelengths]
        spectra_cube = np.zeros(cube_shape)

        # Populate the cube
        for i, params in enumerate(self.param_array):
            # Convert params to indices
            indices = []
            for j, value in enumerate(params):
                # Find index of this value in param_values[j]
                idx = np.where(np.isclose(self.param_values[j], value))[0][0]
                indices.append(idx)

            # Assign spectra to the correct position in the cube
            spectra_cube[tuple(indices)] = self.spectra_array[i]

        return spectra_cube, self.param_values
