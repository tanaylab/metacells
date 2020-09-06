Installation
============

In short: ``pip install metacells``.

**TODO**: Alternatively, use `Anaconda <https://www.anaconda.com/distribution/>`_ and get the conda
packages from the `conda-forge channel <https://anaconda.org/conda-forge/metacells>`_, which
supports both Unix, Mac OS and Windows.

.. todo::

   Provide the Anaconda/Conda packages.

For Unix like systems it is possible to install from source. For Windows this is overly complicated,
and you are recommended to use the binary wheels. Also, this package is meant to be used in the
context of the `scipy <http://scipy.org>`_ framework.

Make sure you have all necessary tools for compilation. In Ubuntu this can be installed using ``sudo
apt-get install build-essential``, please refer to the documentation for your specific system.  Make
sure that not only ``gcc`` is installed, but also ``g++``, as the ``metacells`` package is
programmed in ``C++``.

You can check if all went well by running a variety of tests using ``pytest`` or ``tox``.

