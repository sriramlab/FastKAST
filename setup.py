# setup.py

from setuptools import setup, find_packages

def parse_requirements(filename):
    """Load requirements from a requirements file."""
    with open(filename, "r") as file:
        return [line.strip() for line in file if line and not line.startswith("#")]

setup(
    name="model_x_package",                # Name of the package
    version="0.1.0",                    # Version of the package
    author="Boyang Fu",                
    author_email="fbyang1995@gmail.com", 
    description="This is the package for various epistasis related softwares.",  # A brief description
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sriramlab/FastKAST",  
    packages=find_packages(),         
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
