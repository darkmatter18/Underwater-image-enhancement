from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
    'numpy>=1.18.5',
    'pillow>=7.2.0'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Underwater Image Enhancement'
)
