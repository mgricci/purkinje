from setuptools import setup, find_packages

setup(
    name='purkinje',
    description='A formal passage-of-time model of the cerebellar Purkinje cell',
    version='0.1.0',
    packages=find_packages(),
    #platforms=['mac', 'unix'],
    python_requires='>=3.6',
    install_requires=[
        'jupyter',
        'matplotlib>=3.4.3',
        'numpy>=1.21.3'
    ]
)
