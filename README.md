# matrixCalculator
Requires you to be in a virtual environment which can be activated using
```
. venv/bin/activate
```
Run matrix.py
```
./matrix.py [-h] <command> <matrix...> 
```

## Commands
Single Argument Commands
- det <matrix>
    - Finds determinant of matrix.
- rank <matrix>  
    - Finds rank of matrix.
- inverse <matrix>  
    - Finds inverse of matrix.
- singular <matrix>  
    - Determines whether the matrix is singular.
- square <matrix>  
    - Determines whether the matrix is square.
- rectangular <matrix>  
    - Determines whether the matrix is rectangular.
- null <matrix>  
    - Determines whether the matrix is null.
- identity <matrix>  
    - Determines whether the matrix is the identity matrix.
- transpose <matrix>  
    - Finds the transpose of the matrix.
- adjugate <matrix>  
    - Finds the adjugate of the matrix.
- lower <matrix>
    - Finds the corresponding lower matrix.
- upper <matrix>
    - Finds the corresponding upper matrix.

Multi Argument Commands
- add <matrix> <matrix...>
    - Adds any number of matrices.
- sub <matrix> <matrix...>
    - Subtracts any number of matrices.
- mult <matrix> <matrix...>
    - Multiplies any number of matrices.
- gauss <matrix> <matrix>
    - Performs gaussian elimination to find the solution matrix for two matrices.
    - In the form, Ax = b, it finds x given A and b

## Examples
./matrix.py det [[1,2],[4,5]]
./matrix.py add [[1,2],[3,4]] [[5,6],[7,8]] [[9,0],[1,2]]
./matrix.py gauss "[[1, 2],[3, 4]]" [[3],[6]]

## Note
Be wary of spaces when typing in your matrix as the program sees it as another argument