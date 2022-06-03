#!/usr/bin/env python
from matrix import Matrix

def cli(): # the command line interface
    commands = {
        "help": "list all commands",
        "det": "find the determinant of the matrix",
    }
    functions = {
        "help": "list all commands",
        "det": det,
    }
    print("What do you want to do?")

def det():
    #get matrix input
    matrix.det(input)

if __name__ == "__main__":
    cli()
