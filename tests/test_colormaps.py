from padne import colormaps


class TestUniformColorMap:
    cmap = colormaps.VIRIDIS

    def test_saturation_low(self):
        expected_color = self.cmap.points[0]
        assert self.cmap(-0.1) == expected_color
        assert self.cmap(-100.0) == expected_color
        assert self.cmap(0.0) == expected_color

    def test_saturation_high(self):
        expected_color = self.cmap.points[-1]
        assert self.cmap(1.0) == expected_color
        assert self.cmap(1.1) == expected_color
        assert self.cmap(100.0) == expected_color
        # Just below 1.0 must not index past the end
        assert self.cmap(1.0 - 1e-9) == expected_color
