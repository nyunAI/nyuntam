from setuptools import setup, find_packages
from cli.version import __version__ as version
import os
import sys

readme_file = os.path.join(os.path.dirname(__file__), "README.md")

with open(readme_file, "r") as fh:
    long_description = fh.read()

# required lib
install_requires = ["typer", "docker", "python-dotenv", "PyYAML"]

if sys.version_info.major >= 3 and sys.version_info.minor < 11:
    install_requires.extend(["StrEnum"])

setup(
    name="nyuntam",
    version=version,
    author="NyunAI",
    author_email="contact@nyunai.com",
    description="Nyuntam CLI and Core Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nyunAI/nyuntam.git",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8, <=3.11",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "nyun=cli.cli:app",
        ],
    },
    extras_require={},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
