import unittest
import matrix
Matrix = matrix.Matrix

class TestMatrixGivenRowsCols(unittest.TestCase):

    def test_create_rows(self):
        a = Matrix(2, 1)
        self.assertEqual(a.matrix,[[None],[None]])
    
    def test_set_matrix(self):
        a = Matrix(2,1)
        a.set_matrix([[3],[1]])
        self.assertEqual(a.matrix, [[3],[1]])

    def test_set_matrix_invalid(self):
        a = Matrix(2,1)
        self.assertRaises(ValueError, a.set_matrix, [[3,2],[1,2]])
        

class TestMatrixGivenMat(unittest.TestCase):
    a = Matrix([[1,2,3,4],[5,6,7,8]])

    def test_invalid(self):
        self.assertRaises(ValueError, Matrix, [[1,2,3,4],[5,6]])

    def test_set_matrix(self):
        self.a.set_matrix([[4,3,2,1],[8,7,6,5]])
        self.assertEqual(self.a.matrix, [[4,3,2,1],[8,7,6,5]])

    def test_rows_set_correctly(self):
        self.assertEqual(self.a.rows, 2)
        self.assertEqual(self.a.cols, 4)

    def test_subtraction(self):
        a = Matrix([[1,2,3,4],[1,2,3,4]])
        b = Matrix([[1,2,3,4],[1,2,3,4]])
        self.assertEqual((a-b).matrix, [[0,0,0,0],[0,0,0,0]])

    def test_subtraction_dff_sizes(self):
        a = Matrix([[1,2,3],[1,2,3]])
        b = Matrix([[1,2,3,4],[1,2,3,4]])
        self.assertRaises(ArithmeticError, lambda a,b: a-b, a, b)

    def test_addition(self):
        a = Matrix([[1,2,3,4],[1,2,3,4]])
        b = Matrix([[1,2,3,4],[1,2,3,4]])
        self.assertEqual((a+b).matrix, [[2,4,6,8],[2,4,6,8]])

    def test_addition_dff_sizes(self):
        a = Matrix([[1,2,3],[1,2,3]])
        b = Matrix([[1,2,3,4],[1,2,3,4]])
        self.assertRaises(ArithmeticError, lambda a,b: a+b, a, b)

    def test_determinant(self):
        a = Matrix([[1,2],[3,4]])
        b = Matrix([[1,2]])
        self.assertEqual(-2, a.determinant())
        self.assertRaises(ArithmeticError, b.determinant)
        self.assertEqual(-2, matrixLib.determinant(a))
        self.assertRaises(ArithmeticError, matrixLib.determinant, b)

    def test_same_size(self):
        a = Matrix([[1,2],[3,4]])        
        b = Matrix([[5,6],[7,8]])        
        c = Matrix([[3]])
        self.assertEqual(True, a.same_size(b))
        self.assertEqual(False, a.same_size(c))

    def test_multiplication(self):
        a = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        b = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEqual((a*b).matrix, [[30,36,42],[66,81,96],[102,126,150]])
        self.assertEqual((a*2).matrix, [[2,4,6],[8,10,12],[14,16,18]])

    def test_multiplication_dff_sizes(self):
        a = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        b = Matrix([[1,2],[3,4],[5,6]])
        c = Matrix([[1,2,3,4],[5,6,7,8]])
        self.assertEqual((a*b).matrix, [[22,28],[49,64],[76,100]])
        self.assertRaises(ArithmeticError, lambda a,c: a+c, a, c)

    def test_len(self):
        self.assertEqual(len(self.a), 8)

    def test_get_row(self):
        a = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        b = Matrix([[1,2],[3,4],[5,6]])
        self.assertEqual(a.get_row(0), [1,2,3])
        self.assertEqual(b.get_col(0), [1,3,5])
        self.assertEqual(a.get_diagonal(), [1,5,9])
        self.assertRaises(IndexError, a.get_row, 5)
        self.assertRaises(IndexError, a.get_col, 5)
        self.assertRaises(ArithmeticError, b.get_diagonal)

    def test_transpose(self):
        a = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        b = Matrix([[1,2],[3,4],[5,6]])
        self.assertEqual(a.transpose().matrix, [[1,4,7],[2,5,8],[3,6,9]])
        self.assertEqual(b.transpose().matrix, [[1,3,5],[2,4,6]])

    def test_square(self):
        a = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        b = Matrix([[1,2],[3,4],[5,6]])
        self.assertTrue(a.is_square())
        self.assertFalse(b.is_square())

    def test_matrix_coefficients(self):
        a = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        b = Matrix([[1,2],[3,4],[5,6]])
        # self.assertEqual(a.matrix_coefficients().matrix, [[1,-2,3],[-4,5,-6],[7,-8,9]])
        # self.assertEqual(b.matrix_coefficients().matrix, [[1,-2],[-3,4],[5,-6]])
        self.assertEqual(matrixLib.matrix_coefficients(a).matrix, [[1,-2,3],[-4,5,-6],[7,-8,9]])
        self.assertEqual(matrixLib.matrix_coefficients(b).matrix, [[1,-2],[-3,4],[5,-6]])

    def test_matrix_minors(self):
        a = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        b = Matrix([[1,2],[3,4],[5,6]])
        # self.assertEqual(a.matrix_minors().matrix, [[-3,-6,-3],[-6,-12,-6],[-3,-6,-3]])
        # self.assertRaises(ArithmeticError, b.matrix_minors)
        self.assertEqual(matrixLib.matrix_minors(a).matrix, [[-3,-6,-3],[-6,-12,-6],[-3,-6,-3]])
        self.assertRaises(ArithmeticError, matrixLib.matrix_minors, b)

    def test_adjugate(self):
        a = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        b = Matrix([[1,2],[3,4],[5,6]])
        # self.assertEqual(a.adjugate().matrix, [[-3,6,-3],[6,-12,6],[-3,6,-3]])
        # self.assertRaises(ArithmeticError, b.adjugate)
        self.assertEqual(matrixLib.adjugate(a).matrix, [[-3,6,-3],[6,-12,6],[-3,6,-3]])
        self.assertRaises(ArithmeticError, matrixLib.adjugate, b)

    def test_inverse(self):
        a = Matrix([[4,5],[6,8]])
        b = Matrix([[1,2],[3,4],[5,6]])
        # self.assertEqual(a.inverse().matrix, [[4,-2.5],[-3,2]])
        # self.assertRaises(ArithmeticError, b.inverse)
        self.assertEqual(matrixLib.inverse(a).matrix, [[4,-2.5],[-3,2]])
        self.assertRaises(ArithmeticError, matrixLib.inverse, b)

    def test_is_null(self):
        a = Matrix([[0,0]])
        b = Matrix([[0,0],[0,1]])
        self.assertTrue(a.is_null())
        self.assertFalse(b.is_null())

    def test_is_identity(self):
        a = Matrix([[1,0],[0,1]])
        b = Matrix([[1,1],[0,1]])
        self.assertTrue(a.is_identity())
        self.assertFalse(b.is_identity())

    def test_rank(self):
        a = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        b = Matrix([[1,1],[0,1]])
        c = Matrix([[0,0],[0,0]])
        d = Matrix([[3]])
        self.assertEqual(a.rank(), 2)
        self.assertEqual(b.rank(), 2)
        self.assertEqual(c.rank(), 0)
        self.assertEqual(d.rank(), 1)

    def test_uneven_rank(self):
        a = Matrix([[1,2,3],[4,5,6]])
        b = a.transpose()
        self.assertEqual(a.rank(), 2)
        self.assertEqual(b.rank(), 2)

if __name__ == "__main__":
    unittest.main()
