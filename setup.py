from setuptools import setup

setup(name='vdrnn',
      version='0.1',
      description='implementation of variational recurrent neural networks for learning curve prediction',
      #url='http://github.com/storborg/funniest',
      author='Matilde Gargiani',
      author_email='matildegargiani@gmail.com',
      license='MIT',
      packages=['vdrnn'],
      install_requires=['ConfigSpace', 'torch', 'hpbandster'], 
      zip_safe=False)
