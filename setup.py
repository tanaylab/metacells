import os
import pathlib
import platform
from glob import glob

from setuptools import Extension, find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install

if os.name == "nt":
    raise NotImplementedError("Python metacells does not support native windows.\nInstead, "
                              "install Windows Subsystem for Linux, and metacells within it.")


class CommandMixin(object):
    user_options = [(  #
        'native',
        None,
        'Do not use precompiled (AVX2) wheel, '
        'instead compile on this machine, '
        'targeting its native architecture'
    )]

    def initialize_options(self):
        super().initialize_options()
        self.native = None


class InstallCommand(CommandMixin, install):
    user_options = \
        getattr(install, 'user_options', []) \
        + CommandMixin.user_options


class DevelopCommand(CommandMixin, develop):
    user_options = \
        getattr(develop, 'user_options', []) \
        + CommandMixin.user_options


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

BASE_COMPILE_ARGS = ['-std=c++14', '-ffast-math', '-fassociative-math']
NATIVE_COMPILE_ARGS = ['-march=native', '-mtune=native']
if platform.processor() == 'x86_64':
    WHEEL_COMPILE_ARGS = ['-mavx2']
else:
    WHEEL_COMPILE_ARGS = []

BASE_MACROS = [('ASSERT_LEVEL', 1)]  # 0 for none, 1 for fast, 2 for slow.
NATIVE_MACROS = []
WHEEL_MACROS = [('CHECK_AVX2', 1)]

if str(os.getenv('WHEEL', '')) == '':
    EXTRA_COMPILE_ARGS = BASE_COMPILE_ARGS + NATIVE_COMPILE_ARGS
    DEFINE_MACROS = BASE_MACROS + NATIVE_MACROS
else:
    EXTRA_COMPILE_ARGS = BASE_COMPILE_ARGS + WHEEL_COMPILE_ARGS
    DEFINE_MACROS = BASE_MACROS + WHEEL_MACROS

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
        'Development Status :: 4 - Beta',
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
            extra_compile_args=EXTRA_COMPILE_ARGS,
            define_macros=DEFINE_MACROS,
        ),
    ],
    entry_points={'console_scripts': [
        'metacells_timing=metacells.scripts.timing:main',
    ]},
    packages=find_packages(exclude=['*tests*']),
    python_requires='>=3.7',
    setup_requires=SETUP_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    extras_require={  # TODO: Is this the proper way of expressing these dependencies?
        'test': INSTALL_REQUIRES + TESTS_REQUIRE,
        'develop': INSTALL_REQUIRES + TESTS_REQUIRE + DEVELOP_REQUIRES
    },
    cmdclass={
        'install': InstallCommand,
        'develop': DevelopCommand,
    }
)
