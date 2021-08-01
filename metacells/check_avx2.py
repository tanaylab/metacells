'''
Check for AVX2
--------------

This is only imported on X86_64 machines if we compiled the C++ extension to use AVX2 (when creating
the pre-compiled wheels). If this is run on a non-AVX2 machine, it generates a human-readable error
instead of generating an opaque segmentation fault.
'''

import os
from warnings import warn

try:
    HAS_AVX2 = False
    with (open('/proc/cpuinfo') if os.path.exists('/proc/cpuinfo')
          else os.popen('sysctl -a')) as file:
        for line in file.readlines():
            if 'avx2' in line:
                HAS_AVX2 = True
                break

except BaseException:  # pylint: disable=broad-except
    AVX2_MAYBE_NOT_SUPPORTED = \
        'The metacells precompiled wheel is using AVX2 instructions.\n' \
        "However, AVX2 might not be available on this computer's processors.\n" \
        'Therefore, using the metacells package might cause a segmentation violation.\n' \
        'You can avoid the wheel using: pip install metacells --install-option=--native\n' \
        "This will compile the metacells package on and for this computer's processors."
    warn(AVX2_MAYBE_NOT_SUPPORTED)
    HAS_AVX2 = True

if not HAS_AVX2:
    AVX2_NOT_SUPPORTED = \
        'The metacells precompiled wheel is using AVX2 instructions.\n' \
        "However, AVX2 is not available on this computer's processors.\n" \
        'You can avoid the wheel using: pip install metacells --install-option=--native\n' \
        "This will compile the metacells package on and for this computer's processors."
    raise ImportError(AVX2_NOT_SUPPORTED)
