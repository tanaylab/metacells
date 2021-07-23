'''
Hardware Information
--------------------

Obtain basic information about the hardware we are running from. Yes, this really should be a part
of the standard Python library.

We are only doing this to figure out whether hyper-threading is enabled. Better usage would be to
restrict the amount of parallelization to not exceed the machine's RAM capacity, but the current
divide-and-conquer implementation isn't that clever for that.

This module has been downloaded (mostly) as-is from `qutip
<https://github.com/qutip/qutip/blob/master/qutip/hardware_info.py>`_. It has been slightly tweaked
to shut ``mypy`` and ``pylint`` up.
'''

# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

__all__ = ['hardware_info']

import multiprocessing
import os
import sys
from typing import Any, Dict

import numpy as np


def _mac_hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = dict()
    results: Dict[str, Any] = dict()
    with os.popen('sysctl hw') as file:
        for line in [line.split(':') for line in file.readlines()[1:]]:
            info[line[0].strip(' "').replace(' ', '_').lower().strip('hw.')] = \
                line[1].strip('.\n ')
    results['cpus'] = int(info['physicalcpu'])
    with os.popen('sysctl hw.cpufrequency') as file:
        results['cpu_freq'] = \
            int(float(file.readlines()[0].split(':')[1]) / 1000000)
    results['memsize'] = int(int(info['memsize']) / (1024 ** 2))
    # add OS information
    results['os'] = 'Mac OSX'
    return results


def _linux_hardware_info() -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    # get cpu number
    sockets = 0
    cores_per_socket = 0
    frequency = 0.0
    with open('/proc/cpuinfo') as file:
        for parts in [line.split(':') for line in file.readlines()]:
            if parts[0].strip() == 'physical id':
                sockets = np.maximum(sockets, int(parts[1].strip())+1)
            if parts[0].strip() == 'cpu cores':
                cores_per_socket = int(parts[1].strip())
            if parts[0].strip() == 'cpu MHz':
                frequency = float(parts[1].strip()) / 1000.
    results['cpus'] = int(sockets * cores_per_socket)
    # get cpu frequency directly (bypasses freq scaling)
    try:
        with open('/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq') \
                as file:
            line = file.readlines()[0]
        frequency = float(line.strip('\n')) / 1000000.
    except:  # pylint: disable=bare-except
        pass
    results['cpu_freq'] = frequency

    # get total amount of memory
    mem_info = dict()
    with open('/proc/meminfo') as file:
        for parts in [line.split(':') for line in file.readlines()]:
            mem_info[parts[0]] = parts[1].strip('.\n ').strip('kB')
    results['memsize'] = int(mem_info['MemTotal']) / 1024
    # add OS information
    results['os'] = 'Linux'
    return results


def _freebsd_hardware_info() -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    with os.popen('sysctl -n hw.ncpu') as file:
        results['cpus'] = int(file.readlines()[0])
    with os.popen('sysctl -n dev.cpu.0.freq') as file:
        results['cpu_freq'] = int(file.readlines()[0])
    with os.popen('sysctl -n hw.realmem') as file:
        results['memsize'] = int(file.readlines()[0]) / 1024
    results['os'] = 'FreeBSD'
    return results


def _win_hardware_info() -> Dict[str, Any]:
    try:
        # pylint: disable=import-error,import-outside-toplevel
        from comtypes.client import CoGetObject  # type: ignore
        winmgmts_root = CoGetObject(r'winmgmts:root\cimv2')
        cpus = winmgmts_root.ExecQuery('Select * from Win32_Processor')
        ncpus = 0
        for cpu in cpus:
            ncpus += int(cpu.Properties_['NumberOfCores'].Value)
    except:  # pylint: disable=bare-except
        ncpus = int(multiprocessing.cpu_count())
    return {'os': 'Windows', 'cpus': ncpus}


def hardware_info() -> Dict[str, Any]:
    '''
    Returns a dictionary with basic hardware information about the computer.

    Gives actual number of CPU's in the machine, even when hyperthreading is
    turned on.
    '''
    if sys.platform == 'darwin':
        out = _mac_hardware_info()
    elif sys.platform == 'win32':
        out = _win_hardware_info()
    elif sys.platform in ['linux', 'linux2']:
        out = _linux_hardware_info()
    elif sys.platform.startswith('freebsd'):
        out = _freebsd_hardware_info()
    else:
        out = {}
    return out
