import argparse


def get_opt():
    parser = argparse.ArgumentParser()

    #dataset
    parser.add_argument('--dataset', type=str, default='sst', help='choosing dataset from sst and yelp')

    #model
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha of smmothing')

    return parser.parse_args()
