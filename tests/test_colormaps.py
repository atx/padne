import pytest
from padne import colormaps
# math is used by the implementation, not directly in these tests,
# but good to keep in mind for behavior.

class TestUniformColorMap:
    # Use colormaps.VIRIDIS for all tests as per request
    cmap = colormaps.VIRIDIS

    def test_return_type_and_length_at_0_5(self):
        """Test that the return value for input 0.5 is a 3-tuple of floats."""
        color = self.cmap(0.5)
        assert isinstance(color, tuple), "Color should be a tuple"
        assert len(color) == 3, "Color tuple should have 3 components"
        for component in color:
            # The type hint for UniformColorMap.points specifies floats.
            assert isinstance(component, float), "Color components should be floats"

    def test_saturation_low(self):
        """Test that inputting values below 0.0 properly saturates to the first color."""
        # The first color in the map is the expected saturation point for low values.
        expected_color = self.cmap.points[0]

        assert self.cmap(-0.1) == expected_color, "Value -0.1 should saturate to the first color"
        assert self.cmap(-100.0) == expected_color, "Value -100.0 should saturate to the first color"
        # Also test 0.0 itself, as per implementation logic (i=0, floor(0)=0)
        assert self.cmap(0.0) == expected_color, "Value 0.0 should map to the first color"

    def test_saturation_high(self):
        """Test that inputting values >= 1.0 properly saturates to the last color."""
        # The last color in the map is the expected saturation point for high values.
        expected_color = self.cmap.points[-1]

        # Test 1.0 itself
        # Implementation: i = 1.0 * len; if i >= len: i = len - 1; floor(len-1) = len-1
        assert self.cmap(1.0) == expected_color, "Value 1.0 should saturate to the last color"

        # Test values greater than 1.0
        assert self.cmap(1.1) == expected_color, "Value 1.1 should saturate to the last color"
        assert self.cmap(100.0) == expected_color, "Value 100.0 should saturate to the last color"

        # Test a value very close to 1.0 but less than 1.0
        # This tests the math.floor() behavior for indices like `len(points) - epsilon`.
        # Example: if len(points) is 256, v = 0.999...
        # i = 0.999... * 256 = 255.999...
        # floor(i) = 255, which is index for points[-1]
        # Ensure epsilon is small enough: epsilon <= 1/len(points)
        # 1e-9 is much smaller than 1/256, so this is fine.
        almost_one = 1.0 - 1e-9
        assert self.cmap(almost_one) == expected_color, \
            f"Value {almost_one} (very close to 1.0 from below) should map to the last color"
