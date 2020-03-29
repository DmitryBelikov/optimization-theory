from setuptools import setup, find_packages

setup(name='optimization',
      version='1.0',
      packages=find_packages(),
      install_requires=[
          'scipy==1.4.1',
          'numpy==1.18.2'
      ]
      )
