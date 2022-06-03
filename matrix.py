#!/usr/bin/env python
import argparse
import re
import matrixLib

single_arg = {
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
multi_arg = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mult": lambda x, y: x * y,
    "gauss": lambda x, y: matrixLib.gaussian(x, y),
}

parser = argparse.ArgumentParser(description='Do matrix calculations.')
parser.add_argument('action', choices=list(single_arg.keys())+list(multi_arg.keys()))
parser.add_argument('matrix', help="the matrix to do the operation on", nargs="+")

args = parser.parse_args()

matrices = []

pattern = re.compile("^[\[\]0-9\,\ \.\-\+]*$")
for matrix in args.matrix:
    if not pattern.match(matrix):
        raise TypeError("Matrix contains invalid characters")
    mat = eval(matrix)
    print(mat)
    matrices.append(matrixLib.Matrix(mat))

if args.action in single_arg.keys():
    if len(matrices) != 1:
        raise SyntaxError("Invalid number of arguments")

    func = getattr(matrices[0], single_arg[args.action])
    print(func())
else:
    func = multi_arg[args.action]

    ans = matrices[0]
    for matrix in matrices[1:]:
        ans = func(ans, matrix)
    print(ans)

