"""
Process a ``timing.csv`` file.
"""

import csv
import os
import sys
from argparse import ArgumentParser
from textwrap import dedent
from typing import Dict
from typing import List
from typing import Optional


def main() -> None:
    """
    Process a ``timing.csv`` file.
    """
    parser = ArgumentParser(description="Process a timing.csv file")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", title="commands", help="One of:")

    subparsers.required = True

    combine_parser = subparsers.add_parser(
        "combine",
        help="Combine parallel timing files",
        description=dedent(
            """
        Read a main timings.csv file and all the parallel timing.<map>.<process>.csv files, and
        combine them to a single timing CSV file written to the standard output. In this case, the
        input main timing CSV file name is required (to allow locating the associated parallel
        timing CSV file names).

        The data in the combined file normalizes the data from the parallel map loops such that the
        results show the average elapsed time per process within such loops, but still the total CPU
        time.

        The combined output is in no particular order. Pass it through the sum or flame command
        to obtain a human-readable format.
    """
        ),
    )

    combine_parser.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT",
        help="The optional path to write the output to " "(otherwise, writes to the standard output).",
    )

    combine_parser.add_argument("input", metavar="FILE", help="The path to the timing CSV file.")

    sum_parser = subparsers.add_parser(
        "sum",
        help="Sum the total time for each step",
        description=dedent(
            """
        Read a timings.csv file from the input and write a sum file with one line per
        step containing the sum of the data and the number of invocations to the output.

        The data in the sum file is presented in seconds and billions of instructions, to make it
        easier to follow.

        The output is sorted in descending elapsed time order.

        You can pipe the output through "column -t -s," to make it more legible.
    """
        ),
    )

    sum_parser.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT",
        help="The optional path to write the output to " "(otherwise, writes to the standard output).",
    )

    sum_parser.add_argument(
        "input",
        metavar="FILE",
        nargs="?",
        help="The optional path to the timing CSV file " "(otherwise, reads from the standard input).",
    )

    flame_parser = subparsers.add_parser(
        "flame",
        help="Reformat the data for visualization in flamegraph",
        description=dedent(
            """
        Read a timings.csv file from the input and write a flamegraph file with the
        chosen data (by default, elapsed time). This can be viewed by any flamegraph
        visualization tool such as flameview.
    """
        ),
    )

    flame_parser.add_argument(
        "-f",
        "--focus",
        default="elapsed",
        choices=["elapsed", "cpu", "invocations"],
        help="Which field to focus the flamegraph on.",
    )

    flame_parser.add_argument(
        "-s",
        "--seconds",
        action="store_true",
        help="Output the data in float seconds instead of integer "
        "nanoseconds. Note that most flamegraph viewers can "
        "only handle integer data (flameview can handle float).",
    )

    flame_parser.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT",
        help="The optional path to write the output to " "(otherwise, writes to the standard output).",
    )

    flame_parser.add_argument(
        "input",
        metavar="FILE",
        nargs="?",
        help="The optional path to the timing CSV file " "(otherwise, reads from the standard input).",
    )

    args = parser.parse_args()

    if args.input is not None:
        assert args.input.endswith(".csv")

    if args.command == "combine":
        assert args.input is not None
        _combine_main(args.input, args.output)

    elif args.command == "sum":
        _sum_main(args.input, args.output)

    elif args.command == "flame":
        _flame_main(args.input, args.output, args.focus, args.seconds)


def _combine_main(input_path: str, output_path: Optional[str]) -> None:
    map_index = 0

    if output_path is None:
        output_file = sys.stdout
    else:
        output_file = open(output_path, "w", encoding="utf8")  # pylint: disable=consider-using-with

    for line in open(input_path, "r", encoding="utf8").readlines():  # pylint: disable=consider-using-with
        if ";parallel_map," not in line:
            output_file.write(line)
            continue

        map_index += 1
        line = line.strip()
        fields = line.split(",")
        assert fields[5] == "index"
        assert map_index == int(fields[6])
        assert fields[7] == "processes"
        fields[7] = "expected_processes"
        expected_processes_count = int(fields[8])

        actual_processes_count = expected_processes_count
        process_index = 0
        while True:
            process_index += 1
            process_input = f"{input_path[:-4]}.{map_index}.{process_index}.csv"
            if not os.path.exists(process_input):
                actual_processes_count = process_index
                break

        fields.append("actual_processes")
        fields.append(str(actual_processes_count))
        assert fields[1] == "elapsed_ns"
        fields[2] = str(float(fields[2]) / (actual_processes_count + 1))
        line = ",".join(fields)
        output_file.write(line)
        output_file.write("\n")

        for process_index in range(actual_processes_count):
            process_input = f"{input_path[:-4]}.{map_index}.{process_index}.csv"
            with open(process_input, "r", encoding="utf8") as file:
                for process_line in file.readlines():
                    process_fields = process_line.split(",")
                    assert process_fields[1] == "elapsed_ns"
                    process_fields[2] = str(float(process_fields[2]) / (actual_processes_count + 1))
                    process_line = ",".join(process_fields)
                    output_file.write(process_line)


def _sum_main(input_path: Optional[str], output_path: Optional[str]) -> None:
    data_by_name = _collect_data_by_name(input_path, True)

    total_data: List[float] = []
    for name, data in data_by_name.items():
        while len(total_data) < len(data):
            total_data.append(0)
        for index, datum in enumerate(data):
            total_data[index] += datum

    data_by_name["TOTAL"] = total_data

    if output_path is None:
        output_file = sys.stdout
    else:
        output_file = open(output_path, "w", encoding="utf8")  # pylint: disable=consider-using-with

    fields = ["invocations", "elapsed_s", "cpu_s"]
    for name, data in sorted(data_by_name.items(), key=lambda data: data[1][1], reverse=True):
        text = [name]
        for field, value in zip(fields, data):
            text.append(field)
            if field == "invocations":
                text.append(str(value))
            else:
                text.append(str(value / 1_000_000_000))
        output_file.write(",".join(text))
        output_file.write("\n")


def _flame_main(input_path: Optional[str], output_path: Optional[str], focus: str, seconds: bool) -> None:
    data_by_name = _collect_data_by_name(input_path, False)

    if output_path is None:
        output_file = sys.stdout
    else:
        output_file = open(output_path, "w", encoding="utf8")  # pylint: disable=consider-using-with

    for name, data in data_by_name.items():
        if seconds:
            datum = dict(invocations=data[0], elapsed=data[1] / 1_000_000_000, cpu=data[2] / 1_000_000_000)
        else:
            datum = dict(invocations=data[0], elapsed=int(round(data[1])), cpu=int(round(data[2])))
        html = (
            "Elapsed Time: %.2f<br/>"  # pylint: disable=consider-using-f-string
            "CPU Time: %.2f<br/>"
            "Utilization: %.0f%%<br/>"
            "Invocations: %s<br/>"
            % (datum["elapsed"], datum["cpu"], 100 * datum["cpu"] / datum["elapsed"], datum["invocations"])
        )
        output_file.write(f"{name.replace('.', ';')} {datum[focus]} #{html}\n")


def _collect_data_by_name(input_path: Optional[str], split: bool) -> Dict[str, List[float]]:
    data_by_name: Dict[str, List[float]] = {}

    if input_path is None:
        input_file = sys.stdin
    else:
        input_file = open(input_path, "r", encoding="utf8")  # pylint: disable=consider-using-with

    for row in csv.reader(input_file):
        name = row[0]
        if split:
            name = name.split(";")[-1]
        assert row[1] == "elapsed_ns"
        elapsed_ns = float(row[2])
        assert row[3] == "cpu_ns"
        cpu_ns = float(row[4])

        data = data_by_name.get(name)
        if data is None:
            data = [0, 0, 0]
            data_by_name[name] = data
        data[0] += 1
        data[1] += elapsed_ns
        data[2] += cpu_ns

    return data_by_name


if __name__ == "__main__":
    main()
