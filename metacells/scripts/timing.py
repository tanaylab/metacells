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
    Process a ``timing.csv`` file.
    '''
    parser = ArgumentParser(description='Process a `timing.csv` file')
    subparsers = parser.add_subparsers(dest='command', metavar='COMMAND',
                                       title='commands', help='One of:')

    subparsers.required = True

    _sum_parser = subparsers.add_parser('sum', help='Sum the total time for each step',
                                        description=dedent('''
        Read a `timings.csv` file from the input and write a sum file with one line per
        step containing the sum of the data and the number of invocations to the output.

        The data in the sum file is presented in seconds and billions of instructions, to make it
        easier to follow.

        The output is sorted in descending elapsed time order.

        You can pipe the output through `column -t -s,` to make it more legible.
    '''))

    flame_parser = \
        subparsers.add_parser('flame',
                              help='Reformat the data for visualization in flamegraph',
                              description=dedent('''
        Read a `timings.csv` file from the input and write a flamegraph file with the
        chosen data (by default, `elapsed` time). This can be viewed by any
        flamegraph visualization tool such as flameview.
    '''))

    flame_parser.add_argument('-f', '--focus', default='elapsed',
                              choices=['elapsed', 'cpu', 'invocations'],
                              help='Which field to focus the flamegraph on.')

    flame_parser.add_argument('-s', '--seconds', action='store_true',
                              help='Output the data in float seconds instead of integer '
                                   'nanoseconds. Note that most flamegraph viewers can '
                                   'only handle integer data (flameview can handle float).')

    args = parser.parse_args()

    if args.command == 'sum':
        _sum_main()

    elif args.command == 'flame':
        _flame_main(args.focus, args.seconds)


def _sum_main() -> None:
    data_by_name = _collect_data_by_name(True)

    total_data: List[int] = []
    for name, data in data_by_name.items():
        while len(total_data) < len(data):
            total_data.append(0)
        for index, datum in enumerate(data):
            total_data[index] += datum

    data_by_name['TOTAL'] = total_data

    fields = ['invocations', 'elapsed_s', 'cpu_s']
    for name, data in sorted(data_by_name.items(), key=lambda data: data[1][1], reverse=True):
        text = [name]
        for field, value in zip(fields, data):
            text.append(field)
            if field == 'invocations':
                text.append(str(value))
            else:
                text.append(str(value / 1_000_000_000))
        print(','.join(text))


def _flame_main(focus: str, seconds: bool) -> None:
    data_by_name = _collect_data_by_name(False)

    for name, data in data_by_name.items():
        if seconds:
            datum = dict(invocations=data[0],
                         elapsed=data[1] / 1_000_000_000,
                         cpu=data[2] / 1_000_000_000)
        else:
            datum = dict(invocations=data[0], elapsed=data[1], cpu=data[2])
        html = 'Elapsed Time: %.2f<br/>' \
               'CPU Time: %.2f<br/>' \
               'Utilization: %.0f%%<br/>' \
               'Invocations: %s<br/>' \
               % (datum['elapsed'],
                  datum['cpu'],
                  100 * datum['cpu'] / datum['elapsed'],
                  datum['invocations'])
        print('%s %s #%s' % (name.replace('.', ';'), datum[focus], html))


def _collect_data_by_name(split: bool) -> Dict[str, List[int]]:
    data_by_name: Dict[str, List[int]] = {}

    for row in csv.reader(sys.stdin):
        name = row[0]
        if split:
            name = name.split(';')[-1]
        assert row[1] == 'elapsed_ns'
        elapsed_ns = int(row[2])
        assert row[3] == 'cpu_ns'
        cpu_ns = int(row[4])

        data = data_by_name.get(name)
        if data is None:
            data = [0, 0, 0]
            data_by_name[name] = data
        data[0] += 1
        data[1] += elapsed_ns
        data[2] += cpu_ns

    return data_by_name


if __name__ == '__main__':
    main()
