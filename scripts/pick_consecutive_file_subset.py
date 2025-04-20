import argparse

import sys
from numpy import random


def write_n_consecutive_lines_to_file(input_file: str, output_file: str, num_lines: int, starting_line: int):
    with open(input_file, "r") as istream:
        for _ in range(starting_line - 1):
            istream.readline()

        with open(output_file, "w") as ostream:
            for _ in range(num_lines):
                ostream.write(
                    istream.readline()
                )


def get_file_line_count(file_path: str) -> int:
    with open(file_path, "r") as istream:
        num_lines = sum(1 for _ in istream)

    return num_lines


def main(input_file: str, output_file: str, num_lines: int, starting_line):
    input_file_line_count = get_file_line_count(input_file)
    print("Total lines in {}: {}".format(input_file, input_file_line_count))

    max_starting_line = input_file_line_count - num_lines

    if starting_line is not None:
        if starting_line + num_lines > input_file_line_count:
            print("Error {} + {} > {}. Quitting.".format(starting_line, num_lines, input_file_line_count), file=sys.stderr)
            sys.exit(1)
    else:
        rand_starting_point = random.randint(0, max_starting_line)
        print("Random starting point: {}".format(rand_starting_point))
        starting_line = rand_starting_point

    print("Starting writing lines")
    write_n_consecutive_lines_to_file(
        input_file=input_file,
        output_file=output_file,
        num_lines=num_lines,
        starting_line=starting_line
    )
    print("Finished writing lines")


if __name__ == '__main__':
    program_description = """
        Pick n consecutive lines starting from line m from an input file and write it to an output file
    """
    arg_parse_obj = argparse.ArgumentParser(description=program_description)

    arg_parse_obj.add_argument("input_file", type=str, help="The path to the input file.")
    arg_parse_obj.add_argument("output_file", type=str, help="The path to the output file.")
    arg_parse_obj.add_argument("num_lines", type=int, help="Number of consecutive lines.")
    arg_parse_obj.add_argument("--starting_line", type=int, help="the number of the starting line. "
                                                               "Chosen at random if not provided.",
                               required=False, default=None)

    args = arg_parse_obj.parse_args()

    main(
        input_file=args.input_file,
        output_file=args.output_file,
        num_lines=int(args.num_lines),
        starting_line=int(args.starting_line)
    )
