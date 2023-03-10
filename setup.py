from setuptools import setup, find_packages

setup(
  name = 'sac2021',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  version = '0.0.3',
  license='MIT',
  description = '5th solution for Samsung AI Challenge for Scientific Discovery (2021)',
  author = 'Dohoon Lee',
  author_email = 'dohlee.bioinfo@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/dohlee/samsung-ai-challenge-solution',
  keywords = [
    'artificial intelligence',
    'cheminformatics',
    'molecular property prediction',
    'quantum chemistry',
  ],
  install_requires=[
    'einops>=0.3',
    'rdkit',
    'numpy',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
  ],
)
