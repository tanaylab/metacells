import pathlib
from glob import glob

from setuptools import Extension, find_packages, setup

CWD = pathlib.Path(__file__).parent

README = (CWD / 'README.rst').read_text()

SETUP_REQUIRES = [
    'setuptools_scm',
]

# TODO: Repeated in setup.cfg

INSTALL_REQUIRES = [
    'scanpy',
    'importlib-metadata',
    'numpy',
    'pandas',
    'scipy',
    'python-igraph',
    'threadpoolctl',
]

TESTS_REQUIRE = [
    'pytest',
    'pyyaml',
]

DEVELOP_REQUIRES = [
    'autopep8',
    'isort',
    'mypy',
    'pylint',
    'pylint_strict_informational',
    'sphinx',
    'sphinx_rtd_theme',
    'typing_extensions'
]

setup(
    name='metacells',
    use_scm_version=dict(write_to='metacells/version.py'),
    description='Single-cell RNA Sequencing Analysis',
    long_description=README,
    long_description_content_type='text/x-rst',
    url='https://github.com/tanaylab/metacells.git',
    author='Oren Ben-Kiki',
    author_email='oren@ben-kiki.org',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
    ],
    ext_modules=[
        Extension(  #
            'metacells.extensions',
            include_dirs=['pybind11/include'],
            sources=['metacells/extensions.cpp'],
            extra_compile_args=['-std=c++14',
                                '-march=native', '-mtune=native',
                                '-ffast-math', '-fassociative-math'],
            # extra_link_args=['-lgomp'],
            define_macros=[
                ('ASSERT_LEVEL', 1),  # 0 for none, 1 for fast, 2 for slow.
            ],
        ),
    ],
    entry_points={'console_scripts': [
        'metacells_timing=metacells.scripts.timing:main',
    ]},
    packages=find_packages(),
    python_requires='>=3.7',
    setup_requires=SETUP_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    extras_require={  # TODO: Is this the proper way of expressing these dependencies?
        'test': INSTALL_REQUIRES + TESTS_REQUIRE,
        'develop': INSTALL_REQUIRES + TESTS_REQUIRE + DEVELOP_REQUIRES
    },
)
