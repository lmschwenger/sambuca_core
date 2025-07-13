import unittest
import tempfile
import os
import numpy as np

from sambuca.core.lookup_table import LookUpTable
from sambuca.core.siop_manager import SIOPManager


class TestLookUpTable(unittest.TestCase):
    """Tests for LookUpTable functionality."""

    def setUp(self):
        """Set up test data."""
        # Use default SIOPManager (will try to load from default directory)
        # This makes the test more realistic by using actual SIOP data
        try:
            self.siop_manager = SIOPManager()
        except Exception:
            # If default SIOP data not available, create minimal test data
            self.temp_dir = tempfile.mkdtemp()
            self._create_test_siop_files()
            self.siop_manager = SIOPManager(self.temp_dir)
        
        # Define test wavelengths and parameter ranges
        self.wavelengths = [400, 450, 500, 550, 600]
        self.parameter_ranges = {
            'chl': (0.1, 2.0),
            'cdom': (0.01, 0.5),
        }
        self.fixed_parameters = {
            'nap': 0.1,
            'depth': 10.0,
            'substrate_fraction': 1.0
        }
    
    def _create_test_siop_files(self):
        """Create minimal test SIOP files if default data not available."""
        # Create water absorption file
        water_abs_data = np.column_stack([
            self.wavelengths,
            [0.01, 0.02, 0.05, 0.15, 0.30]
        ])
        np.savetxt(os.path.join(self.temp_dir, 'water_absorption.csv'), 
                   water_abs_data, delimiter=',', 
                   header='wavelength,absorption', comments='')
        
        # Create phytoplankton absorption file
        ph_abs_data = np.column_stack([
            self.wavelengths,
            [0.05, 0.04, 0.02, 0.01, 0.008]
        ])
        np.savetxt(os.path.join(self.temp_dir, 'phytoplankton_absorption.csv'), 
                   ph_abs_data, delimiter=',', 
                   header='wavelength,absorption', comments='')
        
        # Create substrate file
        substrate_data = np.column_stack([
            self.wavelengths,
            [0.1, 0.15, 0.2, 0.25, 0.3]
        ])
        np.savetxt(os.path.join(self.temp_dir, 'sand_substrate.csv'), 
                   substrate_data, delimiter=',', 
                   header='wavelength,reflectance', comments='')

    def tearDown(self):
        """Clean up temporary files."""
        if hasattr(self, 'temp_dir'):
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_lookup_table_creation(self):
        """Test creating a LookUpTable instance."""
        lut = LookUpTable(
            siop_manager=self.siop_manager,
            wavelengths=self.wavelengths,
            parameter_ranges=self.parameter_ranges,
            fixed_parameters=self.fixed_parameters
        )
        
        # Check initialization
        self.assertEqual(lut.wavelengths, self.wavelengths)
        self.assertEqual(lut.parameter_ranges, self.parameter_ranges)
        self.assertEqual(lut.fixed_parameters, self.fixed_parameters)
        self.assertEqual(lut.param_names, ['chl', 'cdom'])
        self.assertEqual(len(lut.bounds), 2)
        self.assertFalse(lut.table_built)
        
        # Check SIOP data was loaded
        self.assertIn('wavelengths', lut.siops)
        self.assertTrue(len(lut.siops) > 1)  # Should have spectral data

    def test_build_table_small_grid(self):
        """Test building a small lookup table."""
        lut = LookUpTable(
            siop_manager=self.siop_manager,
            wavelengths=self.wavelengths,
            parameter_ranges=self.parameter_ranges,
            fixed_parameters=self.fixed_parameters
        )
        
        # Build with small grid for speed
        lut.build_table(grid_size=3, progress_bar=False)
        
        # Check table was built
        self.assertTrue(lut.table_built)
        self.assertEqual(lut.grid_shape, (3, 3))  # 3x3 = 9 combinations (2 variable params)
        self.assertEqual(len(lut.param_array), 9)
        self.assertEqual(lut.spectra_array.shape, (9, 5))  # 9 spectra x 5 wavelengths
        
        # Check parameter values
        self.assertEqual(len(lut.param_values), 2)  # 2 variable parameters
        for param_vals in lut.param_values:
            self.assertEqual(len(param_vals), 3)
        
        # Check that spectra are reasonable (positive values)
        self.assertTrue(np.all(lut.spectra_array >= 0))
        self.assertTrue(np.all(lut.spectra_array < 1))  # Reflectance should be < 1

    def test_build_table_memory_optimized(self):
        """Test building table with memory optimization."""
        lut = LookUpTable(
            siop_manager=self.siop_manager,
            wavelengths=self.wavelengths,
            parameter_ranges=self.parameter_ranges,
            fixed_parameters=self.fixed_parameters
        )
        
        lut.build_table(grid_size=2, memory_optimized=True, progress_bar=False)
        
        self.assertTrue(lut.table_built)
        self.assertFalse(lut.in_memory)
        self.assertEqual(len(lut.points), 0)  # Points dict should be empty

    def test_save_and_load_compressed(self):
        """Test saving and loading with compressed format."""
        lut = LookUpTable(
            siop_manager=self.siop_manager,
            wavelengths=self.wavelengths,
            parameter_ranges=self.parameter_ranges,
            fixed_parameters=self.fixed_parameters
        )
        
        # Build table
        lut.build_table(grid_size=2, progress_bar=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.lut') as tmp_file:
            tmp_filename = tmp_file.name
        
        try:
            lut.save(tmp_filename, compressed=True)
            
            # Check that files were created
            self.assertTrue(os.path.exists(tmp_filename + '_arrays.npz'))
            self.assertTrue(os.path.exists(tmp_filename + '_param_values.npz'))
            self.assertTrue(os.path.exists(tmp_filename + '_attrs'))
            
            # Load the table
            loaded_lut = LookUpTable.load(tmp_filename, self.siop_manager)
            
            # Check that loaded table matches original
            self.assertEqual(loaded_lut.wavelengths, lut.wavelengths)
            self.assertEqual(loaded_lut.parameter_ranges, lut.parameter_ranges)
            self.assertEqual(loaded_lut.param_names, lut.param_names)
            self.assertTrue(loaded_lut.table_built)
            self.assertEqual(loaded_lut.grid_shape, lut.grid_shape)
            
            # Check arrays match
            np.testing.assert_array_equal(loaded_lut.param_array, lut.param_array)
            np.testing.assert_array_equal(loaded_lut.spectra_array, lut.spectra_array)
            
        finally:
            # Clean up
            for suffix in ['_arrays.npz', '_param_values.npz', '_attrs']:
                if os.path.exists(tmp_filename + suffix):
                    os.unlink(tmp_filename + suffix)

    def test_save_and_load_uncompressed(self):
        """Test saving and loading with uncompressed format."""
        lut = LookUpTable(
            siop_manager=self.siop_manager,
            wavelengths=self.wavelengths,
            parameter_ranges=self.parameter_ranges,
            fixed_parameters=self.fixed_parameters
        )
        
        # Build table
        lut.build_table(grid_size=2, progress_bar=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_filename = tmp_file.name
        
        try:
            lut.save(tmp_filename, compressed=False)
            
            # Check that file was created
            self.assertTrue(os.path.exists(tmp_filename))
            
            # Load the table
            loaded_lut = LookUpTable.load(tmp_filename, self.siop_manager)
            
            # Check that loaded table matches original
            self.assertEqual(loaded_lut.wavelengths, lut.wavelengths)
            self.assertEqual(loaded_lut.parameter_ranges, lut.parameter_ranges)
            self.assertTrue(loaded_lut.table_built)
            
            # Check arrays match
            np.testing.assert_array_equal(loaded_lut.param_array, lut.param_array)
            np.testing.assert_array_equal(loaded_lut.spectra_array, lut.spectra_array)
            
        finally:
            # Clean up
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)

    def test_get_spectra_cube(self):
        """Test getting spectra as a parameter cube."""
        lut = LookUpTable(
            siop_manager=self.siop_manager,
            wavelengths=self.wavelengths,
            parameter_ranges=self.parameter_ranges,
            fixed_parameters=self.fixed_parameters
        )
        
        # Build small table
        lut.build_table(grid_size=2, progress_bar=False)
        
        # Get spectra cube
        spectra_cube, param_values = lut.get_spectra_cube()
        
        # Check cube shape: (2, 2, 5) - 2x2 parameter grid, 5 wavelengths
        expected_shape = (2, 2, 5)
        self.assertEqual(spectra_cube.shape, expected_shape)
        
        # Check parameter values
        self.assertEqual(len(param_values), 2)  # 2 variable parameters
        for param_vals in param_values:
            self.assertEqual(len(param_vals), 2)  # 2 points each

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test missing required parameters
        with self.assertRaises(ValueError) as cm:
            LookUpTable(
                siop_manager=self.siop_manager,
                wavelengths=self.wavelengths,
                parameter_ranges={'chl': (0.1, 2.0)},  # Missing other required params
                fixed_parameters={}
            )
        self.assertIn("Missing required parameters", str(cm.exception))
        
        # Test parameter conflicts (same param in both ranges and fixed)
        with self.assertRaises(ValueError) as cm:
            LookUpTable(
                siop_manager=self.siop_manager,
                wavelengths=self.wavelengths,
                parameter_ranges={'chl': (0.1, 2.0)},
                fixed_parameters={'chl': 1.0, 'cdom': 0.1, 'nap': 0.05, 'depth': 10.0, 'substrate_fraction': 1.0}
            )
        self.assertIn("cannot be both variable and fixed", str(cm.exception))

    def test_error_handling(self):
        """Test error handling for various edge cases."""
        lut = LookUpTable(
            siop_manager=self.siop_manager,
            wavelengths=self.wavelengths,
            parameter_ranges={},  # Empty parameter ranges
            fixed_parameters={'chl': 1.0, 'cdom': 0.1, 'nap': 0.05, 'depth': 10.0, 'substrate_fraction': 1.0}
        )
        
        # Should raise error when trying to build with no variable parameters
        with self.assertRaises(ValueError):
            lut.build_table()
        
        # Should raise error when trying to get cube before building
        with self.assertRaises(ValueError):
            lut.get_spectra_cube()

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            LookUpTable.load('nonexistent_file.pkl', self.siop_manager)


if __name__ == '__main__':
    unittest.main()