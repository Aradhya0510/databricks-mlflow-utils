from setuptools import setup, find_packages

setup(
    name='your-package-name',  # Replace with your package name
    version='0.1.0',
    author='Aradhya Chouhan',
    author_email='aradhya0510@gmail.com',
    description='Tools and add-ons for MLflow to facilitate experiment reproduction and replication.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Aradhya0510/databricks-mlflow-utils.git',  # Replace with your GitHub repo URL
    packages=find_packages(),
    install_requires=[
        'mlflow',
        'pandas',
        'scikit-learn',
        # Add other dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Choose your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify your Python version compatibility
)
