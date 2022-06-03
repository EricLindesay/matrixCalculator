#!/usr/bin/env python
import argparse
import re

def det(a):
    print(a)

def rank(a):
    print("rakn: ",a)

def matrix(string):
    # starts and ends with []. Contains floats
    # floats are contained within []
    pass

valid_commands = ['det', 'rank']

parser = argparse.ArgumentParser(description='Do matrix calculations.')
#parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                    help='an integer for the accumulator')
#parser.add_argument('matrix', metavar='N', type=matrix, nargs='+',
#                    help='an integer for the accumulator')
parser.add_argument('action', choices=valid_commands)
parser.add_argument('matrix')

args = parser.parse_args()


print(args.action)

print(f"{args.matrix!r}")
print(args.matrix)

pattern = re.compile("^[\[\]0-9\,\ \.\-\+]*$")
if not pattern.match(args.matrix):
    raise TypeError("Matrix contains invalid characters")
# check args.matrix is valid chars using regex
# eval args.matrix
# do the correct command


print(f"{eval('[[2.3, 39], [93, 20], [2, 3902]]')!r}")

#print(args.rank(args.integers))

