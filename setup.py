import metacells.version
import pathlib

from setuptools import setup
from glob import glob

CWD = pathlib.Path(__file__).parent

README = (CWD / 'README.rst').read_text()

setup(
    name='metacells',
    version=metacells.version.__version__,
    description='Single-cell RNA Sequencing Analysis',
    long_description=README,
    long_description_content_type='text/x-rst',
    url='https://github.com/orenbenkiki/metacells.git',
    author='Oren Ben-Kiki',
    author_email='oren@ben-kiki.org',
    license='MIT',
    license_files='LICENSE.rst',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
    ],
    packages=['metacells'],
    python_requires='>=3.7',
    install_requires=['scipy'],
    tests_require=['pytest'],
    data_files=[('', glob('data/*'))],
)
