"""Setup script for FIFA World Cup 2026 Quiniela Predictor V2."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh.readlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="fifa-world-cup-2026-quiniela-v2",
    version="0.2.0",
    description="Aggressive World Cup 2026 prediction system optimized for quinielas.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DSarceno",
    author_email="dsarceno68@gmail.com",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
