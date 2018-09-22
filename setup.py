from setuptools import setup, find_packages

def _requirements():
    return [name.rstrip() for name in open('requirements.txt').readlines()]

setup(name='seq_gan',
      version='0.0.1',
      description='seq_gan with keras',
      author='Tomoaki Nakamura',
      install_requires=_requirements(),
      packages=find_packages(exclude=('tests', 'docs')),
      url='https://github.com/tyo-yo/SeqGAN',
)
