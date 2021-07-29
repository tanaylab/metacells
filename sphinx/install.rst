Installation
============

In short: ``pip install metacells``. If you do not have ``sudo`` privileges, you might need to ``pip
install --user metacells``. Note that ``metacells`` requires many "heavy" dependencies, most notably
``numpy``, ``pandas``, ``scipy``, ``scanpy``, which ``pip`` should automatically install for you.

Note that ``metacells`` only runs natively on Linux and MacOS. To run it on a Windows computer, you
must activate `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl>`_ and
install ``metacells`` within it.

The metacells package contains extensions written in C++. The ``metacells`` distribution provides
pre-compiled Python wheels for both Linux and MacOS, so installing it using ``pip`` should not
require a C++ compilation step.

Note that for X86 CPUs, these pre-compiled wheels were built to use AVX2, and will not work on older
CPUs which are limited to SSE. Also, these wheels will not make use of any newer instructions (such
as AVX512), even if available. While these wheels may not the perfect match for the machine you are
running on, they are expected to work well for most machines.

To see the native capabilities of your machine, you can ``grep flags /proc/cpuinfo | head -1`` which
will give you a long list of supported CPU features in an arbitrary order, which may include
``sse``, ``avx2``, ``avx512``, etc. You can therefore simply ``grep avx2 /proc/cpuinfo | head -1``
to test whether AVX2 is/not supported by your machine.

You can avoid installing the pre-compiled wheel by running ``pip install metacells
--install-option='--native'``. This will force ``pip`` to compile the C++ extensions locally on your
machine, optimizing for its native capabilities, whatever these may be. However, this requires you
to have a C++ compiler installed (either ``g++`` or ``clang``), and it will take much longer to
complete the installation.

