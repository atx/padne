
import pytest

from padne import problem as p


class TestNetwork:

    def test_triangle_construction(self):
        n_a = p.NodeID()
        n_b = p.NodeID()
        n_c = p.NodeID()

        # Check that we get a new object every time
        assert n_a != n_b
        assert n_b != n_c
        assert n_a != n_c
        assert n_a == n_a

        r_ab = p.Resistor(n_a, n_b, 1)
        r_bc = p.Resistor(n_b, n_c, 1)
        r_cd = p.Resistor(n_c, n_a, 1)

        net = p.Network([], [r_ab, r_bc, r_cd])
        assert not net.has_source
        # Order not guaranteed
        assert set(net.nodes) == {n_a, n_b, n_c}
        assert [net.nodes[n] for n in net.nodes] == [0, 1, 2]


class TestResistor:

    def test_valid_resistance(self):
        n_a = p.NodeID()
        n_b = p.NodeID()
        r = p.Resistor(n_a, n_b, 100.0)
        assert r.resistance == 100.0

    def test_invalid_resistance_zero(self):
        n_a = p.NodeID()
        n_b = p.NodeID()
        with pytest.raises(ValueError, match="Resistance must be positive"):
            p.Resistor(n_a, n_b, 0.0)

    def test_invalid_resistance_negative(self):
        n_a = p.NodeID()
        n_b = p.NodeID()
        with pytest.raises(ValueError, match="Resistance must be positive"):
            p.Resistor(n_a, n_b, -1.0)
