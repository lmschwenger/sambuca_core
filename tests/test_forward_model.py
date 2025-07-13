import unittest
import numpy as np

from sambuca.core.forward_model import forward_model, ForwardModelResults


class TestForwardModelIntegration(unittest.TestCase):
    """Integration tests for forward model functionality."""

    def setUp(self):
        """Set up test data for forward model tests."""
        self.num_bands = 5
        self.wavelengths = [400, 450, 500, 550, 600]
        self.a_water = [0.00635, 0.0145, 0.0596, 0.2885, 0.5318]
        self.a_ph_star = [0.0624, 0.0436, 0.0201, 0.0123, 0.0082]
        self.substrate1 = [0.15, 0.18, 0.22, 0.25, 0.28]
        
        # Basic model parameters
        self.chl = 0.5
        self.cdom = 0.1
        self.nap = 0.05
        self.depth = 10.0

    def test_forward_model_basic_execution(self):
        """Test that forward model executes without errors."""
        result = forward_model(
            chl=self.chl,
            cdom=self.cdom,
            nap=self.nap,
            depth=self.depth,
            substrate1=self.substrate1,
            wavelengths=self.wavelengths,
            a_water=self.a_water,
            a_ph_star=self.a_ph_star,
            num_bands=self.num_bands
        )
        
        self.assertIsInstance(result, ForwardModelResults)

    def test_forward_model_output_shapes(self):
        """Test that forward model outputs have correct shapes."""
        result = forward_model(
            chl=self.chl,
            cdom=self.cdom,
            nap=self.nap,
            depth=self.depth,
            substrate1=self.substrate1,
            wavelengths=self.wavelengths,
            a_water=self.a_water,
            a_ph_star=self.a_ph_star,
            num_bands=self.num_bands
        )
        
        # Check all arrays have correct length
        self.assertEqual(len(result.rrs), self.num_bands)
        self.assertEqual(len(result.rrsdp), self.num_bands)
        self.assertEqual(len(result.r_0_minus), self.num_bands)
        self.assertEqual(len(result.rdp_0_minus), self.num_bands)
        self.assertEqual(len(result.kd), self.num_bands)
        self.assertEqual(len(result.a), self.num_bands)
        self.assertEqual(len(result.bb), self.num_bands)

    def test_forward_model_with_two_substrates(self):
        """Test forward model with two substrates."""
        substrate2 = [0.10, 0.12, 0.14, 0.16, 0.18]
        substrate_fraction = 0.7
        
        result = forward_model(
            chl=self.chl,
            cdom=self.cdom,
            nap=self.nap,
            depth=self.depth,
            substrate1=self.substrate1,
            wavelengths=self.wavelengths,
            a_water=self.a_water,
            a_ph_star=self.a_ph_star,
            num_bands=self.num_bands,
            substrate2=substrate2,
            substrate_fraction=substrate_fraction
        )
        
        self.assertIsInstance(result, ForwardModelResults)
        self.assertEqual(len(result.r_substratum), self.num_bands)

    def test_forward_model_zero_concentrations(self):
        """Test forward model with zero concentrations."""
        result = forward_model(
            chl=0.0,
            cdom=0.0,
            nap=0.0,
            depth=self.depth,
            substrate1=self.substrate1,
            wavelengths=self.wavelengths,
            a_water=self.a_water,
            a_ph_star=self.a_ph_star,
            num_bands=self.num_bands
        )
        
        # With zero concentrations, phytoplankton components should be zero
        self.assertTrue(np.allclose(result.a_ph, 0.0))
        self.assertTrue(np.allclose(result.a_cdom, 0.0))
        self.assertTrue(np.allclose(result.a_nap, 0.0))
        self.assertTrue(np.allclose(result.bb_ph, 0.0))
        self.assertTrue(np.allclose(result.bb_nap, 0.0))

    def test_forward_model_deep_water(self):
        """Test forward model with very deep water (optically deep)."""
        result = forward_model(
            chl=self.chl,
            cdom=self.cdom,
            nap=self.nap,
            depth=1000.0,  # Very deep
            substrate1=self.substrate1,
            wavelengths=self.wavelengths,
            a_water=self.a_water,
            a_ph_star=self.a_ph_star,
            num_bands=self.num_bands
        )
        
        # For very deep water, rrs should approach rrsdp
        self.assertTrue(np.allclose(result.rrs, result.rrsdp, rtol=1e-2))

    def test_forward_model_input_validation(self):
        """Test forward model input validation."""
        # Test mismatched array lengths
        with self.assertRaises(AssertionError):
            forward_model(
                chl=self.chl,
                cdom=self.cdom,
                nap=self.nap,
                depth=self.depth,
                substrate1=[0.15, 0.18],  # Wrong length
                wavelengths=self.wavelengths,
                a_water=self.a_water,
                a_ph_star=self.a_ph_star,
                num_bands=self.num_bands
            )

    def test_forward_model_positive_outputs(self):
        """Test that forward model produces positive outputs where expected."""
        result = forward_model(
            chl=self.chl,
            cdom=self.cdom,
            nap=self.nap,
            depth=self.depth,
            substrate1=self.substrate1,
            wavelengths=self.wavelengths,
            a_water=self.a_water,
            a_ph_star=self.a_ph_star,
            num_bands=self.num_bands
        )
        
        # All these should be positive
        self.assertTrue(np.all(result.rrs >= 0))
        self.assertTrue(np.all(result.rrsdp >= 0))
        self.assertTrue(np.all(result.kd > 0))
        self.assertTrue(np.all(result.a > 0))
        self.assertTrue(np.all(result.bb > 0))


if __name__ == '__main__':
    unittest.main()