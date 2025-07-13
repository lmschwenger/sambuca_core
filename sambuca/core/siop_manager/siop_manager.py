"""Management of Spectral Inherent Optical Properties (SIOPs) for different sensors."""

import csv
import os
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d


class SIOPManager:
    """Manager for Spectral Inherent Optical Properties.

    This class manages the loading, storage, and interpolation of spectral libraries
    to custom wavelength configurations.
    """
    
    DEFAULT_SIOPS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'siops')

    def __init__(self, siop_directory: Optional[str] = None):
        """Initialize the SIOP manager.

        Args:
            siop_directory: Optional path to directory containing spectral libraries.
                If not provided, uses DEFAULT_SIOPS_DIR. Libraries will be loaded immediately.
        """
        self.siop_directory = siop_directory or self.DEFAULT_SIOPS_DIR
        self.raw_libraries = {}  # Original spectral libraries

        self.load_libraries(self.siop_directory)

    def load_libraries(self, directory: str) -> None:
        """Load all spectral libraries from a directory.

        Args:
            directory: Path to directory containing spectral libraries.
        """
        self.siop_directory = directory

        # Direct CSV loading approach
        self.raw_libraries = self._load_csv_libraries(directory)

        print(f"Loaded {len(self.raw_libraries)} spectral libraries from {directory}")

    def _load_csv_libraries(self, directory: str) -> Dict[str, Tuple[NDArray, NDArray]]:
        """Load CSV spectral libraries directly.

        Args:
            directory: Path to directory containing CSV files.

        Returns:
            Dictionary of (wavelengths, values) tuples.
        """
        libraries = {}

        # Walk through directory and subdirectories
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    filepath = os.path.join(root, file)
                    try:
                        # Extract library type from directory and file name
                        rel_path = os.path.relpath(filepath, directory)
                        parts = rel_path.split(os.sep)

                        # Create library name from file name and parent directory
                        file_base = os.path.splitext(parts[-1])[0]

                        # Load CSV using built-in csv module
                        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
                            # Detect delimiter
                            sample = csvfile.read(1024)
                            csvfile.seek(0)
                            sniffer = csv.Sniffer()
                            delimiter = sniffer.sniff(sample).delimiter
                            
                            # Read CSV data
                            reader = csv.reader(csvfile, delimiter=delimiter)
                            rows = list(reader)
                        
                        if len(rows) < 2:
                            print(f"Skipping {file}: insufficient data")
                            continue
                        
                        # Get header and data
                        header = rows[0]
                        data_rows = rows[1:]
                        
                        # Convert to numpy arrays
                        data_array = np.array(data_rows, dtype=float)
                        
                        # Check number of columns
                        if len(header) == 2:
                            # Simple two-column format
                            library_name = file_base.lower()

                            libraries[library_name] = (
                                data_array[:, 0],  # wavelengths
                                data_array[:, 1]   # values
                            )
                            print(f"Loaded {library_name} from {rel_path}")
                        else:
                            # Multi-column format
                            wavelengths = data_array[:, 0]

                            for col_idx in range(1, len(header)):
                                col_name = header[col_idx]
                                library_name = f"{file_base}_{col_name}".lower()
                                libraries[library_name] = (
                                    wavelengths,
                                    data_array[:, col_idx]
                                )
                                print(f"Loaded {library_name} from {rel_path}")
                    except Exception as e:
                        print(f"Error loading {file}: {e}")

        return libraries

    def get_siops_for_wavelengths(self, target_wavelengths: Union[List[float], NDArray]) -> Dict[str, Any]:
        """Get all SIOPs interpolated to match specified wavelengths.

        Args:
            target_wavelengths: Target wavelengths for interpolation.

        Returns:
            Dictionary with interpolated SIOPs for the specified wavelengths.
        """
        if not isinstance(target_wavelengths, np.ndarray):
            target_wavelengths = np.array(target_wavelengths)

        result = {
            'wavelengths': target_wavelengths,
            'num_bands': len(target_wavelengths)
        }

        # Process each spectral library
        for name, (src_wavelengths, src_values) in self.raw_libraries.items():
            # Check if wavelength ranges overlap enough
            if min(src_wavelengths) > max(target_wavelengths) or \
                    max(src_wavelengths) < min(target_wavelengths):
                print(f"Warning: Spectral library '{name}' does not cover the target wavelength range")
                continue

            # Create interpolator
            interpolator = interp1d(
                src_wavelengths,
                src_values,
                bounds_error=False,
                fill_value="extrapolate"
            )

            # Interpolate to target wavelengths
            result[name] = interpolator(target_wavelengths)

        return result

    def list_available_libraries(self) -> List[str]:
        """List all available spectral library names.

        Returns:
            List of spectral library names.
        """
        return list(self.raw_libraries.keys())

    def get_common_library_types(self) -> Dict[str, List[str]]:
        """Group libraries by common types (absorption, backscatter, substrate).

        Returns:
            Dictionary with grouped library names.
        """
        types = {
            'absorption': [],
            'backscatter': [],
            'substrate': [],
            'other': []
        }

        for name in self.raw_libraries.keys():
            if 'absorption' in name:
                types['absorption'].append(name)
            elif 'backscatter' in name:
                types['backscatter'].append(name)
            elif 'substrate' in name:
                types['substrate'].append(name)
            else:
                types['other'].append(name)

        return types

