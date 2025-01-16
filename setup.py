from setuptools import setup, find_packages

setup(
    name='mantis_shrimp',
    version='0.1',
    url='https://github.com/pnnl/mantis_shrimp.git',
    author='Andrew Engel',
    author_email='andrew.engel@pnnl.gov',
    description='Photometric Redshift Estimation using Computer Vision',
    packages=find_packages(),
    include_package_data=True,
    package_data={"mantis_shrimp": ["configs/*.json",
                                    "MODELS_final/*.pt",
                                    "MODELS_final/calpit_stats/*npy"
                                    "dustmaps/planck/*.fits",
                                    "dustmaps/csfd/*.fits"],},
)
