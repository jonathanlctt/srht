from setuptools import setup, find_packages

setup(
  name = 'srht',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'Subsampled randomized Hadamard transform',
  author = 'Jonathan Lacotte',
  author_email = 'lacotte@stanford.edu',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/jonathanlctt/srht',
  keywords = [
    'random projection',
    'least squares',
    'hadamard',
    'optimizers'
  ],
  install_requires=[
    'torch>=1.6',
    'numpy'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
