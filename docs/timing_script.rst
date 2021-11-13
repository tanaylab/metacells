Timing
======

.. automodule:: metacells.scripts.timing

Usage
-----

.. code::

    usage: metacells_timing.py [-h] COMMAND ...

    Process a timing.csv file

    optional arguments:
      -h, --help  show this help message and exit

    commands:
      COMMAND     One of:
        combine   Combine parallel timing files
        sum       Sum the total time for each step
        flame     Reformat the data for visualization in flamegraph

Combine Command
---------------

.. code::

    usage: metacells_timing.py combine [-h] [-o OUTPUT] FILE

    Read a main timings.csv file and all the parallel timing.<map>.<process>.csv
    files, and combine them to a single timing CSV file written to the standard
    output. In this case, the input main timing CSV file name is required (to
    allow locating the associated parallel timing CSV file names). The data in the
    combined file normalizes the data from the parallel map loops such that the
    results show the average elapsed time per process within such loops, but still
    the total CPU time. The combined output is in no particular order. Pass it
    through the sum or flame command to obtain a human-readable format.

    positional arguments:
      FILE                  The path to the timing CSV file.

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTPUT, --output OUTPUT
                            The optional path to write the output to (otherwise,
                            writes to the standard output).

Flame Command
-------------

.. code::

    usage: metacells_timing.py sum [-h] [-o OUTPUT] [FILE]

    Read a timings.csv file from the input and write a sum file with one line per
    step containing the sum of the data and the number of invocations to the
    output. The data in the sum file is presented in seconds and billions of
    instructions, to make it easier to follow. The output is sorted in descending
    elapsed time order. You can pipe the output through "column -t -s," to make it
    more legible.

    positional arguments:
      FILE                  The optional path to the timing CSV file (otherwise,
                            reads from the standard input).

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTPUT, --output OUTPUT
                            The optional path to write the output to (otherwise,
                            writes to the standard output).

Sum Command
-----------

.. code::

    usage: metacells_timing.py flame [-h] [-f {elapsed,cpu,invocations}] [-s] [-o OUTPUT]
                           [FILE]

    Read a timings.csv file from the input and write a flamegraph file with the
    chosen data (by default, elapsed time). This can be viewed by any flamegraph
    visualization tool such as flameview.

    positional arguments:
      FILE                  The optional path to the timing CSV file (otherwise,
                            reads from the standard input).

    optional arguments:
      -h, --help            show this help message and exit
      -f {elapsed,cpu,invocations}, --focus {elapsed,cpu,invocations}
                            Which field to focus the flamegraph on.
      -s, --seconds         Output the data in float seconds instead of integer
                            nanoseconds. Note that most flamegraph viewers can
                            only handle integer data (flameview can handle float).
      -o OUTPUT, --output OUTPUT
                            The optional path to write the output to (otherwise,
                            writes to the standard output).
