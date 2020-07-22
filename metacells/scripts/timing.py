'''
Script for processing a ``timing.csv`` file.

.. todo::

    Proper script documentation.
'''

import csv
import sys
from argparse import ArgumentParser
from textwrap import dedent
from typing import Dict, List


def main() -> None:
    '''
    Process ``timing.csv`` a file.
    '''
    parser = ArgumentParser(description=dedent('''
        Read a ``timings.csv`` file from the input and write a sum file with one line per step
        containing the number of invocations and the sum of the data to the output.

        The data in the sum file is presented in seconds and billions of instructions, to make it
        easier to follow.

        The output is sorted in descending elapsed time order.

        You can pipe the output through ``column -t -s,`` to make it more legible.
    '''))

    parser.parse_args()

    data_by_name: Dict[str, List[float]] = {}

    for row in csv.reader(sys.stdin):
        name = row[0]
        assert row[1] == 'elapsed_ns'
        elapsed_ns = int(row[2])
        assert row[3] == 'cpu_ns'
        cpu_ns = int(row[4])
        instructions = int(row[6]) if row[5] == 'instructions' else None

        data = data_by_name.get(name)
        if data is None:
            if instructions is None:
                data = [0, 0, 0]
            else:
                data = [0, 0, 0, 0]
            data_by_name[name] = data
        data[0] += 1
        data[1] += elapsed_ns / 1_000_000_000
        data[2] += cpu_ns / 1_000_000_000
        if instructions is not None:
            data[3] += instructions / 1_000_000_000

    total_data: List[float] = []
    for name, data in data_by_name.items():
        while len(total_data) < len(data):
            total_data.append(0)
        for index, datum in enumerate(data):
            total_data[index] += datum

    data_by_name['TOTAL'] = total_data

    fields = ['invocations', 'elapsed_s', 'cpu_s', 'instructions_G']
    for name, data in sorted(data_by_name.items(), key=lambda data: data[1][1], reverse=True):
        text = [name]
        for field, value in zip(fields, data):
            text.append(field)
            text.append(str(value))
        print(','.join(text))


if __name__ == '__main__':
    main()
