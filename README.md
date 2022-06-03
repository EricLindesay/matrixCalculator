# matrixCalculator
Requires you to be in a virtual environment which can be activated using
```
. venv/bin/activate
```
Run matrix.py
```
./matrix.py <command> <matrix...> 
```

## Commands
Single Argument Commands
- det  
- rank  
- inverse  
- singular  
- square  
- rectangular  
- null  
- identity  
- transpose  
- adjugate  

Multi Argument Commands
- add
- sub
- mult

## Example
./matrix.py det [[1,2],[4,5]]
./matrix.py add [[1,2],[3,4]] [[5,6],[7,8]] [[9,0],[1,2]]
./matrix.py gauss "[[1, 2],[3, 4]]" [[3],[6]]

## Note
Be wary of spaces when typing in your matrix as the program sees it as another argument