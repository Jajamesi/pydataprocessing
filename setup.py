from setuptools import setup, find_packages

setup(
    name='dataprocessing',
    version='0.0.1',
    description='Tools to process SPSS sav databases',
    long_description_content_type="text/markdown",
    license='Jajames',
    packages=find_packages(),
    author='Alex Zhironkin',
    author_email='zhironkinalexandr@gmail.com',
    keywords=['Sav', 'SPSS', 'dataprocessing'],
    install_requires = [
        'pandas>=2.1.0',
        'savReaderWriter>=3.4.0',
        'numpy>=1.25.0',
        'thefuzz>=0.20.0',
        'streamlit>=1.27.2',
        'weightipy>=0.3.3',
        'pillow>=11.0.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.11',

