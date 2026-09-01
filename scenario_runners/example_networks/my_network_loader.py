"""
my_network_loader.py
====================
Stub network loader module for the `source: function` plugin example
(custom_function.yaml).

The contract is a ZERO-ARGUMENT function returning a pandapowerNet:

    def get_network() -> pandapowerNet

network_plugin.py imports this file by path (importlib file-location
machinery, same as the controller plugin) and calls the function named by
the YAML's `function:` key.  Replace the body with your own construction
logic — a parametric feeder generator, a network stitched from several
JSON exports, a synthetic stress case, etc.

Demonstration: loads the CIGRE MV benchmark network with PV and wind DERs
(15 buses, 9 DERs) — the same network the built-in OPF scenario uses.
"""

import pandapower.networks as pn


def get_network():
    """Return the CIGRE MV benchmark network with PV + wind DERs."""
    return pn.create_cigre_network_mv(with_der="pv_wind")
