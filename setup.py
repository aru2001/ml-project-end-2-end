from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    with open(file_path, 'r') as file:
        requirements = file.readlines()

    
    return [req.strip() for req in requirements if req.strip() and not req.startswith('#') and req.strip() != HYPEN_E_DOT]
setup(
    name='ml_project',
    version='0.1.0',
    author='Arti Sikhwal',
    author_email='arti.sikhwal.2001@gmail.com',
    description='A machine learning project setup',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)