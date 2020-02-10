'''
	Filter json file
	Author: Sandesh Timilsina
	Date: 02/06/2020

        - Filters aspect category
        - Filters based on target i.e. targeted or untargeted
'''

import os
import sys
import argparse


def main(argv):
    parser =  argparser.ArgumentParser(add_help=Truem description=('Filter json file'))
    parser.add_argument('-file', default="../../brat/data/nepsa/all_channels.json", help='input json file path')
    parser.add_argument('-a', default="['GENERAL','PROFANITY','VIOLENCE','FEEDBACK','OUTOFSCOPE']", help='aspect category filter')
    parser.add_argument('-t', default="[targeted, untargeted]", help='targeted or untargeted filter')

    args = parser.parse_args(argv)
    input_json_file = args.file
    aspect_filter = args.a
    target_filter = args.t

if __name__ == "__main__":
    main(sys.argv[1:])
