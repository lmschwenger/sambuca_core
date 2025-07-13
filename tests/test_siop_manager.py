import unittest

from sambuca.core.siop_manager import SIOPManager


class TestSIOPManagerIntegration(unittest.TestCase):
    """Integration tests for SIOPManager functionality."""

    def test_siop_manager_initialization(self):
        """Test SIOPManager initialization with valid SIOP files."""
        sm = SIOPManager()

        # Check if SIOPManager initializes with default SIOP files
        self.assertTrue(sm.raw_libraries != {})

    def test_siop_manager_get_siops_for_wavelengths(self):
        sm = SIOPManager()

        """Test getting SIOP files for specific wavelengths."""
        wavelengths = [400, 500, 600]
        siops = sm.get_siops_for_wavelengths(wavelengths)

        # Check if the returned SIOPs match the requested wavelengths
        self.assertTrue(all(w in siops['wavelengths'] for w in wavelengths))