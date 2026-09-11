# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup


setup(
    name="emg2qwerty",
    version="0.2.0",
    description=(
        "Conformer frontend and beam-aware language-model correction "
        "for sEMG typing transcription."
    ),
    author="sEMG Beam-Aware Transcription contributors",
    packages=find_packages(),
    python_requires=">=3.10,<3.11",
    install_requires=[
        # The CUDA-index options and exact runtime snapshot live in
        # requirements.txt. Install that file before using editable mode.
    ],
)
