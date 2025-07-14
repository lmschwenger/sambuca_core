import unittest
import tempfile
import os
import numpy as np

from sambuca.core.inversion_handler import InversionHandler, InversionResult
from sambuca.core.lookup_table import LookUpTable, ParameterType
from sambuca.core.siop_manager import SIOPManager


class TestInversionHandler(unittest.TestCase):
    """Tests for InversionHandler functionality."""

    def setUp(self):
        """Set up test data."""
        # Use default SIOPManager or create test data
        try:
            self.siop_manager = SIOPManager()
        except Exception:
            # If default SIOP data not available, create minimal test data
            self.temp_dir = tempfile.mkdtemp()
            self._create_test_siop_files()
            self.siop_manager = SIOPManager(self.temp_dir)
        
        # Define test parameters
        self.wavelengths = [400, 450, 500, 550, 600]
        self.options = {
            'chl': ParameterType.RANGE(0.1, 2.0, 5),
            'cdom': ParameterType.RANGE(0.01, 0.5, 5),
            'nap': ParameterType.FIXED(0.1),
            'depth': ParameterType.FIXED(10.0),
            'substrate_fraction': ParameterType.FIXED(1.0)
        }
        
        # Keep old format for backward compatibility where needed
        self.parameter_ranges = {
            'chl': (0.1, 2.0),
            'cdom': (0.01, 0.5),
        }
        self.fixed_parameters = {
            'nap': 0.1,
            'depth': 10.0,
            'substrate_fraction': 1.0
        }
        
        # Create and build lookup table
        self.lut = LookUpTable(
            siop_manager=self.siop_manager,
            wavelengths=self.wavelengths,
            options=self.options
        )
        self.lut.build_table(progress_bar=False)  # Grid size now in options
        
        # Create inversion handler
        self.handler = InversionHandler()
    
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

    def test_inversion_handler_creation(self):
        """Test creating an InversionHandler instance."""
        handler = InversionHandler()
        self.assertIsInstance(handler, InversionHandler)

    def test_invert_single_pixel_image(self):
        """Test inverting a single pixel image."""
        # Create a 1x1x5 test image (single pixel)
        test_image = np.random.rand(1, 1, 5) * 0.1  # Low reflectance values
        
        result = self.handler.invert_image_from_lookup_table(
            lookup_table=self.lut,
            observed_image=test_image,
            metric='euclidean'
        )
        
        # Check result structure
        self.assertIsInstance(result, InversionResult)
        self.assertEqual(len(result.parameters), 2)  # chl and cdom
        self.assertIn('chl', result.parameters)
        self.assertIn('cdom', result.parameters)
        
        # Check array shapes
        self.assertEqual(result.parameters['chl'].shape, (1, 1))
        self.assertEqual(result.parameters['cdom'].shape, (1, 1))
        self.assertEqual(result.errors.shape, (1, 1))
        self.assertEqual(result.modeled_spectra.shape, (1, 1, 5))
        
        # Check that values are not NaN (pixel should be processed)
        self.assertFalse(np.isnan(result.parameters['chl'][0, 0]))
        self.assertFalse(np.isnan(result.parameters['cdom'][0, 0]))
        self.assertFalse(np.isnan(result.errors[0, 0]))

    def test_invert_small_image(self):
        """Test inverting a small multi-pixel image."""
        # Create a 3x3x5 test image
        height, width, bands = 3, 3, 5
        test_image = np.random.rand(height, width, bands) * 0.1
        
        result = self.handler.invert_image_from_lookup_table(
            lookup_table=self.lut,
            observed_image=test_image,
            metric='euclidean'
        )
        
        # Check shapes
        self.assertEqual(result.parameters['chl'].shape, (height, width))
        self.assertEqual(result.parameters['cdom'].shape, (height, width))
        self.assertEqual(result.errors.shape, (height, width))
        self.assertEqual(result.modeled_spectra.shape, (height, width, bands))
        
        # Check metadata
        self.assertEqual(result.metadata['n_valid_pixels'], height * width)
        self.assertEqual(result.metadata['n_total_pixels'], height * width)
        self.assertEqual(result.metadata['image_shape'], (height, width, bands))

    def test_invert_with_mask(self):
        """Test inverting an image with a mask."""
        # Create a 4x4x5 test image
        height, width, bands = 4, 4, 5
        test_image = np.random.rand(height, width, bands) * 0.1
        
        # Create a mask that excludes some pixels
        mask = np.array([
            [True,  True,  False, False],
            [True,  True,  False, False],
            [False, False, True,  True],
            [False, False, True,  True]
        ])
        
        result = self.handler.invert_image_from_lookup_table(
            lookup_table=self.lut,
            observed_image=test_image,
            metric='euclidean',
            mask=mask
        )
        
        # Check that masked pixels have NaN values
        for i in range(height):
            for j in range(width):
                if mask[i, j]:
                    # Valid pixel should have real values
                    self.assertFalse(np.isnan(result.parameters['chl'][i, j]))
                    self.assertFalse(np.isnan(result.errors[i, j]))
                else:
                    # Masked pixel should have NaN values
                    self.assertTrue(np.isnan(result.parameters['chl'][i, j]))
                    self.assertTrue(np.isnan(result.errors[i, j]))
        
        # Check metadata
        self.assertEqual(result.metadata['n_valid_pixels'], np.sum(mask))

    def test_different_metrics(self):
        """Test different distance metrics."""
        # Create a 2x2x5 test image
        test_image = np.random.rand(2, 2, 5) * 0.1
        
        # Test different metrics
        for metric in ['euclidean', 'rmse', 'sam']:
            with self.subTest(metric=metric):
                result = self.handler.invert_image_from_lookup_table(
                    lookup_table=self.lut,
                    observed_image=test_image,
                    metric=metric
                )
                
                self.assertIsInstance(result, InversionResult)
                self.assertEqual(result.metadata['metric'], metric)
                
                # Check that we got valid results
                self.assertFalse(np.any(np.isnan(result.parameters['chl'])))
                self.assertFalse(np.any(np.isnan(result.errors)))

    def test_kdtree_vs_brute_force(self):
        """Test that KD-tree and brute force give same results for euclidean metric."""
        # Create a small test image
        test_image = np.random.rand(2, 2, 5) * 0.1
        
        # Test with KD-tree
        result_kdtree = self.handler.invert_image_from_lookup_table(
            lookup_table=self.lut,
            observed_image=test_image,
            metric='euclidean',
            use_kdtree=True
        )
        
        # Test without KD-tree
        result_brute = self.handler.invert_image_from_lookup_table(
            lookup_table=self.lut,
            observed_image=test_image,
            metric='euclidean',
            use_kdtree=False
        )
        
        # Results should be very similar (allowing for small numerical differences)
        np.testing.assert_allclose(result_kdtree.parameters['chl'], 
                                   result_brute.parameters['chl'], rtol=1e-10)
        np.testing.assert_allclose(result_kdtree.parameters['cdom'], 
                                   result_brute.parameters['cdom'], rtol=1e-10)

    def test_chunked_processing(self):
        """Test processing with different chunk sizes."""
        # Create a larger test image
        test_image = np.random.rand(5, 5, 5) * 0.1
        
        # Test with different chunk sizes
        for chunk_size in [1, 5, 25]:
            with self.subTest(chunk_size=chunk_size):
                result = self.handler.invert_image_from_lookup_table(
                    lookup_table=self.lut,
                    observed_image=test_image,
                    metric='euclidean',
                    chunk_size=chunk_size
                )
                
                self.assertEqual(result.metadata['chunk_size'], chunk_size)
                self.assertEqual(result.metadata['n_valid_pixels'], 25)

    def test_parameter_bounds_validation(self):
        """Test that inverted parameters are within expected bounds."""
        # Create test image
        test_image = np.random.rand(3, 3, 5) * 0.1
        
        result = self.handler.invert_image_from_lookup_table(
            lookup_table=self.lut,
            observed_image=test_image,
            metric='euclidean'
        )
        
        # Check that parameters are within the bounds used to build the LUT
        chl_min, chl_max = self.parameter_ranges['chl']
        cdom_min, cdom_max = self.parameter_ranges['cdom']
        
        chl_values = result.parameters['chl'][~np.isnan(result.parameters['chl'])]
        cdom_values = result.parameters['cdom'][~np.isnan(result.parameters['cdom'])]
        
        self.assertTrue(np.all(chl_values >= chl_min))
        self.assertTrue(np.all(chl_values <= chl_max))
        self.assertTrue(np.all(cdom_values >= cdom_min))
        self.assertTrue(np.all(cdom_values <= cdom_max))

    def test_error_handling(self):
        """Test error handling for various edge cases."""
        test_image = np.random.rand(2, 2, 5) * 0.1
        
        # Test with unbuilt lookup table
        unbuilt_lut = LookUpTable(
            siop_manager=self.siop_manager,
            wavelengths=self.wavelengths,
            options=self.options
        )
        
        with self.assertRaises(ValueError) as cm:
            self.handler.invert_image_from_lookup_table(unbuilt_lut, test_image)
        self.assertIn("must be built", str(cm.exception))
        
        # Test with wrong number of bands
        wrong_bands_image = np.random.rand(2, 2, 3) * 0.1
        
        with self.assertRaises(ValueError) as cm:
            self.handler.invert_image_from_lookup_table(self.lut, wrong_bands_image)
        self.assertIn("bands but lookup table expects", str(cm.exception))
        
        # Test with wrong mask shape
        wrong_mask = np.ones((3, 3), dtype=bool)  # Wrong shape for 2x2 image
        
        with self.assertRaises(ValueError) as cm:
            self.handler.invert_image_from_lookup_table(
                self.lut, test_image, mask=wrong_mask
            )
        self.assertIn("Mask shape", str(cm.exception))
        
        # Test with all-False mask
        all_false_mask = np.zeros((2, 2), dtype=bool)
        
        with self.assertRaises(ValueError) as cm:
            self.handler.invert_image_from_lookup_table(
                self.lut, test_image, mask=all_false_mask
            )
        self.assertIn("No valid pixels", str(cm.exception))

    def test_timing_information(self):
        """Test that timing information is provided."""
        test_image = np.random.rand(300, 300, 5) * 0.1
        
        result = self.handler.invert_image_from_lookup_table(
            lookup_table=self.lut,
            observed_image=test_image,
            metric='euclidean'
        )
        
        # Check timing information exists
        self.assertIn('total', result.timing)
        self.assertIn('per_pixel', result.timing)
        self.assertGreater(result.timing['total'], 0)
        self.assertGreater(result.timing['per_pixel'], 0)

    def test_unknown_metric(self):
        """Test error handling for unknown metrics."""
        test_image = np.random.rand(2, 2, 5) * 0.1
        
        with self.assertRaises(ValueError) as cm:
            self.handler.invert_image_from_lookup_table(
                self.lut, test_image, metric='unknown_metric'
            )
        self.assertIn("Unknown metric", str(cm.exception))


if __name__ == '__main__':
    unittest.main()