from numbers import Number
from multipledispatch import dispatch
from time import time
from itertools import combinations

class Matrix():
    matrix = []
    rows = 0
    cols = 0

    @dispatch(int, int)
    def __init__(self, rows: int, cols: int):
        self.matrix = [[None for _ in range(cols)] for _ in range(rows)]
        self.rows = rows
        self.cols = cols

    @dispatch(list)
    def __init__(self, matrix: list):
        self.set_matrix(matrix)
        self.rows = self.num_rows(matrix)
        self.cols = self.num_cols(matrix)

    def basic_loop(self, func) -> object:
        sub_matrix = []
        for r_ind, row in enumerate(self.matrix):
            new_row = []
            for c_ind, val in enumerate(row):
                new_val = func(r_ind, row, c_ind, val)
                if new_val != None:
                    new_row.append(new_val)
            sub_matrix.append(new_row)
        return Matrix(sub_matrix)

    def set_matrix(self, matrix: list) -> None:
        if self.valid_matrix(matrix):
            given_rows = self.num_rows(matrix)
            given_cols = self.num_cols(matrix)
            if self.rows:
                if self.rows != given_rows:
                    raise ValueError(f"Your matrix is {self.rows}x{self.cols} but you input a matrix with {given_rows}x{given_cols}")
            
            if self.cols:
                if self.cols != given_cols:
                    raise ValueError(f"Your matrix is {self.rows}x{self.cols} but you input a matrix with {given_rows}x{given_cols}")

            self.matrix = matrix
        else:
            raise ValueError("Inconsistent number of columns in each row")

    def num_rows(self, matrix: list) -> int:
        return len(matrix)

    def num_cols(self, matrix: list) -> int:
        if len(matrix) > 0:
            return len(matrix[0])
        return 0

    def __add__(self, otherMatrix: object) -> object:
        if not self.same_size(otherMatrix):
            raise ArithmeticError("The matrices you are trying to add are not the same size")

        add = lambda a, b: a+b
        return self.add_sub(otherMatrix, add)

    def __sub__(self, otherMatrix: object) -> object:
        if not self.same_size(otherMatrix):
            raise ArithmeticError("The matrices you are trying to subtract are not the same size")

        sub = lambda a, b: a-b
        return self.add_sub(otherMatrix, sub)

    def add_sub(self, otherMatrix: object, func) -> object:
        a = self.matrix
        b = otherMatrix.matrix      
        c = Matrix(self.rows, self.cols)
        for row_ind, row in enumerate(a):
            for col_ind, elem in enumerate(row):
                c.matrix[row_ind][col_ind] = func(elem, b[row_ind][col_ind])
        return c        

    def same_size(self, b: object) -> bool:
        if self.rows == b.rows:
            if self.cols == b.cols:
                return True
        return False

    @dispatch(Number)
    def __mul__(self, scalar: int) -> object:
        scalar_multiply = lambda a,b,c,val: val*scalar
        return self.basic_loop(scalar_multiply)
    
    @dispatch(object)
    def __mul__(self, otherMatrix: object) -> object:
        if self.cols != otherMatrix.rows:
            raise AttributeError(f"You cannot multiply a {self.rows}x{self.cols} matrix with a {otherMatrix.rows}x{otherMatrix.cols}")
        
        sub_matrix = []
        for r_i, row in enumerate(self.matrix):
            new_row = []
            for c_i in range(len(row)):
                if c_i > otherMatrix.cols-1:
                    continue
                sum = self.row_times_col(row, otherMatrix.get_col(c_i))
                new_row.append(sum)
            sub_matrix.append(new_row)
        return Matrix(sub_matrix)
        
    def row_times_col(self, row: list, col: list) -> float:
        sum = 0
        for r_i, val in enumerate(row):
            sum += val * col[r_i]
        return sum

    def __len__(self) -> int:
        return self.rows*self.cols

    def __str__(self) -> str:
        string = "["
        for ind, row in enumerate(self.matrix):
            if ind > 0:
                string += " "
            string += self.get_row_string(row)

            if ind < len(self.matrix)-1:
               string += "\n" # add a newline on all except the last.

        string += "]"
        return string

    def get_row_string(self, row) -> str:
        string = "["
        for ind, col in enumerate(row):
            string += str(col)
            if ind < len(row)-1:
                string += ", "
        return string+"]"

    def valid_matrix(self, matrix) -> bool:
        cols = -1
        for row in matrix:
            count = len(row)
            if cols == -1:
                cols = count
            elif count != cols:
                return False
        return True

    def get_row(self, index: int) -> list:
        return self.matrix[index]
    
    def get_col(self, index: int) -> list:
        if index > self.cols:
            raise IndexError("list index out of range")
        return [x for row in self.matrix for ind, x in enumerate(row) if ind == index]

    def get_diagonal(self) -> list:
        if not self.is_square():
            raise ArithmeticError("A non-square matrix has no diagonal")
        
        diag = [col for i_r, row in enumerate(self.matrix) for i_c, col in enumerate(row) if i_r == i_c]
        return diag
        
    def transpose(self) -> object:
        transpose_mat = []
        for i in range(self.cols):
            transpose_mat.append(self.get_col(i))
        
        aT = Matrix(transpose_mat)
        return aT

    def is_square(self) -> bool:
        return self.rows == self.cols

    def is_rectangular(self) -> bool:
        return self.rows != self.cols

    @dispatch()
    def determinant(self) -> float:
        if not self.is_square():
            raise ArithmeticError("Cannot find the determinant of a non-square matrix")
        return self.find_determinant(self.matrix)
    
    @dispatch(list)
    def determinant(self, matrix: list) -> float:
        if len(matrix) != len(matrix[0]):
            raise ArithmeticError("Cannot find the determinant of a non-square matrix")
        return self.find_determinant(matrix)
        
    def find_determinant(self, matrix: list) -> float:
        if len(matrix) == 1:
            return matrix[0][0]

        det = 0
        for i in range(len(matrix)):
            sign = -1 if i%2 else 1
            multiplier = matrix[0][i]
            sub_matrix = self.find_sub_matrix(matrix, 0, i)
            det += sign * multiplier * self.find_determinant(sub_matrix)
        return det

    @dispatch(int, int)
    def find_sub_matrix(self, row_index: int, col_index: int) -> list:
        sub_matrix = []
        for r_i, row in enumerate(self.matrix):
            if r_i == row_index:
                continue
            temp_mat = [val for c_i, val in enumerate(row) if c_i != col_index] 
            sub_matrix.append(temp_mat)
        return sub_matrix

    @dispatch(list, list)
    def find_sub_matrix(self, row_index: list, col_index: list) -> list:
        sub_matrix = []
        for r_i, row in enumerate(self.matrix):
            if r_i in row_index:
                continue
            temp_mat = [val for c_i, val in enumerate(row) if not c_i in col_index] 
            sub_matrix.append(temp_mat)
        return sub_matrix

    @dispatch(list, int, int)
    def find_sub_matrix(self, matrix: list, row_index: int, col_index:int) -> list:
        sub_matrix = []
        for r_i, row in enumerate(matrix):
            if r_i == row_index:
                continue
            temp_mat = [val for c_i, val in enumerate(row) if c_i != col_index] 
            sub_matrix.append(temp_mat)
        return sub_matrix

    def matrix_coefficients(self) -> object:
        calculation = lambda r_i, _, c_i, val: ((-1)**((r_i+c_i)%2)) * val
        return self.basic_loop(calculation)
    
    def matrix_minors(self) -> object:
        if not self.is_square():
            raise ArithmeticError("Cannot find the matrix of minors from a non-square matrix")
        find_sub_det = lambda r_i, row, c_i, col: self.find_determinant(self.find_sub_matrix(r_i, c_i))
        return self.basic_loop(find_sub_det)


    def adjugate(self) -> object:
        if not self.is_square():
            raise ArithmeticError("Cannot find the adjugate from a non-square matrix")
        b = self.transpose()
        b = b.matrix_coefficients()
        b = b.matrix_minors()
        return b

    def inverse(self) -> object:
        if not self.is_square():
            raise ArithmeticError("Cannot find the inverse of a non-square matrix")

        det = self.determinant()
        if det == 0:
            raise ArithmeticError("Determinant is zero, you cannot find the inverse of a singular matrix")
        inverse = self.adjugate() * (1/det)
        return inverse
    
    def is_singular(self) -> bool:
        return self.determinant() == 0

    def is_null(self) -> bool:
        for row in self.matrix:
            for val in row:
                if val != 0:
                    return False
        return True

    def is_identity(self) -> bool:
        for r_i, row in enumerate(self.matrix):
            for c_i, val in enumerate(row):
                if r_i == c_i:
                    if val != 1:
                        return False
                else:
                    if val != 0:
                        return False
        return True

    def rank(self) -> int:
        # get all square matrices formed from self.matrix.
        # then call inner_rank on all of those.
        # and return the max of those. 
        # The get_sub_matrix always returns a list with 1 less row and 1 less column.
        # So if you input a square matrix, a square matrix is returned.
        ranks = 0
        if self.is_rectangular():
            matrices = [x.matrix for x in self.get_square_submatrices()]
        else:
            matrices = [self.matrix]
        
        ranks = map(self.inner_rank, matrices)
        return max(ranks)

    def get_square_submatrices(self) -> list:
        smallest = min(self.rows, self.cols)
        largest = max(self.rows, self.cols)
        submatrices = []
        to_ignore = combinations(range(largest), largest-smallest)
        for val in to_ignore:
            val = list(val)
            if largest == self.rows:
                submatrices.append(Matrix(self.find_sub_matrix(val, [-1])))
            elif largest == self.cols:
                submatrices.append(Matrix(self.find_sub_matrix([-1], val)))

        return submatrices

    def inner_rank(self, matrix:list) -> int:
        det = self.determinant(matrix)  #shouldn't be self, should be matrix
        if det != 0:
            return len(matrix)
        if len(matrix) == 1:
            return self.determinant(matrix)

        rank = 0
        for r_i, row in enumerate(self.matrix):
            for c_i, val in enumerate(row):
                sub_rank = self.inner_rank(self.find_sub_matrix(matrix, r_i, c_i))
                if sub_rank > rank:
                    rank = sub_rank
        
        return rank

    def LU_factorise(self) -> object:
        u = [row[:] for row in self.matrix]
        l = identity(self.rows).matrix
        if self.rows != self.cols:
            raise ArithmeticError("Passed in matrix must be square")

        for i in range(1, self.cols): # loop through each row (except first)
            for offset in range(0, self.cols-i): # loop through the remaining rows (row-1) times
                coeff = u[i+offset][i-1]/u[i-1][i-1]
                for j in range(i-1, self.cols): # update each value with the new value
                    u[i+offset][j] = u[i+offset][j] - coeff*u[i-1][j]
                l[i+offset][i-1] = coeff
        return Matrix(l), Matrix(u)

    def upper(self) -> object:
        _, u = self.LU_factorise()
        return u
    
    def lower(self) -> object:
        l, _ = self.LU_factorise()
        return l

def inverse(matrix: Matrix) -> Matrix:
    if not matrix.is_square():
        raise ArithmeticError("Cannot find the inverse of a non-square matrix")

    det = matrix.determinant()
    if det == 0:
        raise ArithmeticError("Determinant is zero, you cannot find the inverse of a singular matrix")
    inverse = adjugate(matrix) * (1/det)
    return inverse
    
def adjugate(matrix: Matrix) -> Matrix:
    if not matrix.is_square():
        raise ArithmeticError("Cannot find the adjugate from a non-square matrix")
    b = matrix.transpose()
    b = matrix_coefficients(b)
    b = matrix_minors(b)
    return b

def matrix_coefficients(matrix: Matrix) -> Matrix:
    calculation = lambda r_i, _, c_i, val: ((-1)**((r_i+c_i)%2)) * val
    return matrix.basic_loop(calculation)
    
def matrix_minors(matrix: Matrix) -> Matrix:
    if not matrix.is_square():
        raise ArithmeticError("Cannot find the matrix of minors from a non-square matrix")
    find_sub_det = lambda r_i, row, c_i, col: matrix.find_determinant(matrix.find_sub_matrix(r_i, c_i))
    return matrix.basic_loop(find_sub_det)

def determinant(matrix: Matrix) -> float:
    if not matrix.is_square():
        raise ArithmeticError("Cannot find the determinant of a non-square matrix")
    return matrix.find_determinant(matrix.matrix)

def rank(matrix: Matrix) -> float:
    return matrix.rank()

def gaussian(A: Matrix, B: Matrix) -> Matrix:
    a = A.matrix
    b = B.matrix

    # if A.rows > A.cols, just treat it as a square matrix
    if A.cols != B.rows:
        raise TypeError("Invalid matrix dimensions")
    
    for i in range(1, A.cols): # loop through each row (except first)
        for offset in range(0, A.cols-i): # loop through the remaining rows
            coeff = a[i+offset][i-1]/a[i-1][i-1]
            for j in range(i-1, A.cols): # update each value with the new value
                a[i+offset][j] = a[i+offset][j] - coeff*a[i-1][j]
            b[i+offset][0] = b[i+offset][0] - coeff*b[i-1][0]

    # now we have the gaussian eliminated form of the matrices
    # now solve it

    sol = []
    for _ in range(B.rows):
        sol.append([0])
    
    for i in range(A.cols-1, -1, -1):
        ans = b[i][0]
        for j in range(A.cols-1, i, -1):
            ans -= sol[j][0]*a[i][j]
        
        sol[i][0] = ans/a[i][i]
        if sol[i][0] < 0.000000001:
            sol[i][0] = 0

    return Matrix(sol)

def identity(size: int) -> Matrix:
    mat = []
    for i in range(size):
        row = []
        for j in range(size):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        mat.append(row)

    return Matrix(mat)

# is linearly dependent
# LU factorisation
# gaussian elim
if __name__ == "__main__":
    print("This file contains the matrix class. You should instead open something else.")
    start_time = time()
    a = Matrix([[1,2,3],[4,5,6],[9,8,9]])
    print(a.upper())
    print(a.lower())
    print(f"Time taken: {time()-start_time}")
