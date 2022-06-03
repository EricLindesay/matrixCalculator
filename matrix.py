#!/usr/bin/env python
import argparse
import re
import matrixLib

commands = {
    'det': 'determinant',
    'rank': 'rank',
    'inverse': 'inverse',
    'singular': 'is_singular',
    'square': 'is_square',
    'rectangular': 'is_rectangular',
    'null': 'is_null',
    'identity': 'is_identity',
    'transpose': 'transpose',
    'adjugate': 'adjugate',
}

parser = argparse.ArgumentParser(description='Do matrix calculations.')
parser.add_argument('action', choices=commands.keys())
parser.add_argument('matrix', help="the matrix to do the operation on")

args = parser.parse_args()

pattern = re.compile("^[\[\]0-9\,\ \.\-\+]*$")
if not pattern.match(args.matrix):
    raise TypeError("Matrix contains invalid characters")

mat_list = eval(args.matrix)
mat = matrixLib.Matrix(mat_list)

func = getattr(mat, commands[args.action])
print(func())

