"""
hiltest — HIL Testbed master test suite package.

Run:
    python -m hiltest                          # all sections
    python -m hiltest --section volt_var_control
    python -m hiltest --section sensitivity_coordinator --only cigre
    python -m hiltest --section volt_var_control --arduino-port /dev/ttyACM0
    python -m hiltest --section sensitivity_coordinator --only-hw --arduino-port COM3
"""

__version__ = "0.1.0"
