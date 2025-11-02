from typing import List
from setuptools import find_packages, setup

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name="AirbnbPricePrediction",
    version="0.1.0",
    author="Nitesh (NETIZEN-11)",
    author_email="nitesh@example.com",
    description="End-to-end ML pipeline + Flask UI for predicting Airbnb nightly prices.",
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
