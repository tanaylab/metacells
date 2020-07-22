'''
Script for processing a ``timing.csv`` file.

.. todo::

    Proper script documentation.
'''

import csv
import sys
from argparse import ArgumentParser
from textwrap import dedent
from typing import Dict, List, Optional, Tuple, Union

import numpy as np  # type: ignore
import yaml


def main() -> None:
    '''
    Process ``timing.csv`` a file.
    '''
    parser = ArgumentParser(description='Process a timing.csv file')
    subparsers = parser.add_subparsers(dest='command', metavar='COMMAND',
                                       title='commands', help='One of:')

    _sum_parser = subparsers.add_parser('sum', help='Sum the total time for each step',
                                        description=dedent('''
        Read a ``timings.csv`` file from the input and write a sum file with one line per
        step containing the sum of the data and the number of invocations to the output.

        The data in the sum file is presented in seconds and billions of instructions, to make it
        easier to follow.

        The output is sorted in descending elapsed time order.

        You can pipe the output through ``column -t -s,`` to make it more legible.
    '''))

    _calibrate_parser = \
        subparsers.add_parser('calibrate',
                              help='Calibrate estimators of used instructions for parallel code',
                              description=dedent('''
        Read a ``timings.csv`` file from the input and write a CSV file with the "best"
        coefficients for predicting the number of instructions that will be used for
        each invocation of a step. This is used by the parallel loop functions to pick
        the minimal size of invocation batches, below which it no longer makes sense to
        run invocations in parallel.

        The output is sorted by step name.

        You can pipe the output through ``column -t -s,`` to make it more legible.
    '''))

    args = parser.parse_args()

    if args.command == 'sum':
        _sum_main()

    elif args.command == 'calibrate':
        _calibrare_main()


def _sum_main() -> None:
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


def _calibrare_main() -> None:
    data_by_name = _collect_calibration_data()
    results = _compute_calibration_results(data_by_name)
    print(yaml.dump(results))


def _collect_calibration_data() -> Dict[str,
                                        Tuple[str,
                                              Dict[str,
                                                   Tuple[List[float], List[float]]]]]:
    data_by_name: Dict[str, Tuple[str,
                                  Dict[str, Tuple[List[float], List[float]]]]] = {}

    for row in csv.reader(sys.stdin):
        name = row[0]

        if row[5] != 'instructions':
            continue
        instructions = int(row[6])

        factors: Dict[str, Optional[str]] = \
            dict(n=None, m=None, complexity=None, variant=None)
        for index in range(7, len(row), 2):
            field = row[index]
            if field in factors:
                factors[field] = row[index + 1]

        complexity = factors['complexity']

        if complexity is None:
            continue

        variant = factors['variant']
        m = float(factors['m'] or '-1')
        n = float(factors['n'] or '-1')
        assert variant is not None
        assert m > 0
        assert n > 0

        if complexity == 'n':
            size = n
        if complexity == 'n_log_n':
            size = float(n * np.log2(n))
        else:
            assert complexity in ['n', 'n_log_n']

        data_by_variant = data_by_name.get(name)
        if data_by_variant is None:
            data_by_variant = data_by_name[name] = (complexity, {})

        data = data_by_variant[1].get(variant)
        if data is None:
            data = data_by_variant[1][variant] = ([], [])

        data[0].append(instructions / m)
        data[1].append(size)

    return data_by_name


def _compute_calibration_results(
    data_by_name: Dict[str, Tuple[str, Dict[str, Tuple[List[float], List[float]]]]],
) -> Dict[str, Dict[str, Dict[str, Union[str, float]]]]:
    results_by_name: Dict[str, Dict[str, Dict[str, Union[str, float]]]] = {}

    for name, data_by_variant in data_by_name.items():
        complexity = data_by_variant[0]
        results_by_variant = results_by_name[name] = {}
        for variant, data in data_by_variant[1].items():
            vector = np.array(data[0])
            matrix = np.empty((len(vector), 2))
            matrix[:, 0] = 1
            matrix[:, 1] = data[1]
            coefficients = np.linalg.lstsq(matrix, vector.T, rcond=None)[0]
            residuals = np.matmul(matrix, coefficients) - vector.T
            relative_absolute_error = \
                np.linalg.norm(residuals) / np.linalg.norm(vector)
            results_by_variant[variant] = \
                dict(complexity=complexity,
                     constant=float(coefficients[0]),
                     factor=float(coefficients[1]),
                     relative_absolute_error=float(relative_absolute_error))

    return results_by_name


if __name__ == '__main__':
    main()
