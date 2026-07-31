import argparse

from rae_stage1_rec_common import get_args_parser, main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "RAEv2 DINOv3-L K=1/7/23 TokBench reconstruction",
        parents=[get_args_parser("raev2")],
    )
    main(parser.parse_args())
