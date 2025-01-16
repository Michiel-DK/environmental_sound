from setuptools import setup, find_packages

# Load requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='environmental_sound',
    version='0.1.0',
    #author='Your Name',  # Replace with your name
    #author_email='your.email@example.com',  # Replace with your email
    description='A package for processing and analyzing environmental sound data.',
    #long_description=open('README.md').read(),
    #long_description_content_type='text/markdown',
    #url='https://github.com/yourusername/environmental_sound',  # Replace with your repository URL
    packages=find_packages(),
    install_requires=requirements,
    # classifiers=[
    #     'Programming Language :: Python :: 3',
    #     'License :: OSI Approved :: MIT License',  # Replace with your chosen license
    #     'Operating System :: OS Independent',
    # ],
    #python_requires='>=3.7',  # Specify the minimum Python version
    include_package_data=True,  # Include additional files specified in MANIFEST.in
)