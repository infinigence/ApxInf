"""Optional CPU acceleration for the Python processor package."""
import sys
from setuptools import Extension, setup

flags = ["/std:c++17", "/O2", "/fp:strict"] if sys.platform == "win32" else [
    "-std=c++17", "-O3", "-ffp-contract=off",
]
setup(ext_modules=[Extension(
    "apxinf.processors._resize_native",
    ["apxinf/processors/native/resize.cpp"],
    language="c++", extra_compile_args=flags, optional=True,
)])
