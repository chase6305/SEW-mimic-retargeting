"""Optional native-extension build configuration."""

from __future__ import annotations

import os
import sys

from setuptools import setup

ext_modules = []
if os.getenv("SEW_MIMIC_BUILD_CPP", "0") == "1":
    from pybind11.setup_helpers import Pybind11Extension

    ext_modules.append(
        Pybind11Extension(
            "sew_mimic._sew_mimic_cpp",
            ["src/cpp/sew_mimic_cpp.cpp"],
            cxx_std=17,
            extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O3"],
        )
    )

setup(ext_modules=ext_modules)
