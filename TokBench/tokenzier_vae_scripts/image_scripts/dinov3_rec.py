import argparse

from rae_stage1_rec_common import get_args_parser, main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DINOv3-L K1 TokBench reconstruction",
        parents=[get_args_parser("dinov3")],
    )
    main(parser.parse_args())
