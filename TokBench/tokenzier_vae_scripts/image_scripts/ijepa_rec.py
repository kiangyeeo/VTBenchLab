import argparse

from rae_stage1_rec_common import get_args_parser, main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "I-JEPA-H K1 TokBench reconstruction",
        parents=[get_args_parser("ijepa")],
    )
    main(parser.parse_args())
