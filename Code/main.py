import argparse
import preprocessing
import tensorflow as tf
import constants

def parse_args():
    """
    Argparser to allow to run different things if needed
    """
    parser = argparse.ArgumentParser(description='3Sigma Final Project')
    parser.add_argument("--",
                        help="",
                        type=int,
                        default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    

if __name__ == "__main__":
    main()
