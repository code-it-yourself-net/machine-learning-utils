// Machine Learning Utils
// File name: Matrix.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.Utils;

public class Matrix
{
    private const string NumberOfColumnsMustBeEqualToNumberOfColumnsMsg = "The number of columns of the first matrix must be equal to the number of columns of the second matrix.";
    private const string NumberOfRowsMustBeEqualToNumberOfRowsMsg = "The number of rows of the first matrix must be equal to the number of rows of the second matrix.";
    private const string NumberOfColumnsMustBeEqualToNumberOfRowsMsg = "The number of columns of the first matrix must be equal to the number of rows of the second matrix.";

    private readonly Array _array;

    /// <summary>
    /// Initializes a new instance of the <see cref="Matrix"/> class with the specified array.
    /// </summary>
    /// <param name="array">The array representing the matrix.</param>
    /// <remarks>
    /// A new instance of the <see cref="Matrix"/> class is filled with zeros.
    /// </remarks>
    public Matrix(Array array)
    {
        _array = array;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Matrix"/> class with the specified number of rows and columns.
    /// </summary>
    /// <param name="rows">The number of rows in the matrix.</param>
    /// <param name="columns">The number of columns in the matrix.</param>
    /// <remarks>
    /// A new instance of the <see cref="Matrix"/> class is filled with zeros.
    /// </remarks>
    public Matrix(int rows, int columns)
    {
        _array = Array.CreateInstance(typeof(float), rows, columns);
    }

    public Array Array => _array;

    /// <summary>
    /// Implicitly converts a <see cref="Matrix"/> to an <see cref="Array"/>.
    /// </summary>
    /// <param name="matrix">The <see cref="Matrix"/> to convert.</param>
    /// <returns>The converted <see cref="Array"/>.</returns>
    public static implicit operator Array(Matrix matrix) => matrix.Array;

    #region Zeros, Ones, and Random

    /// <summary>
    /// Creates a new matrix filled with zeros, with the same dimensions as the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix used to determine the dimensions of the new matrix.</param>
    /// <returns>A new matrix filled with zeros.</returns>
    public static Matrix Zeros(Matrix matrix)
    {
        (int rows, int columns) = GetDimensions(matrix);
        return Zeros(rows, columns);
    }

    /// <summary>
    /// Creates a new matrix filled with zeros, with the specified number of rows and columns.
    /// </summary>
    /// <param name="rows">The number of rows in the matrix.</param>
    /// <param name="columns">The number of columns in the matrix.</param>
    /// <returns>A new matrix filled with zeros.</returns>
    public static Matrix Zeros(int rows, int columns) => new(rows, columns);

    /// <summary>
    /// Creates a new matrix filled with ones, with the same dimensions as the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix used to determine the dimensions of the new matrix.</param>
    /// <returns>A new matrix filled with ones.</returns>
    public static Matrix Ones(Matrix matrix)
    {
        (int rows, int columns) = GetDimensions(matrix);
        return Ones(rows, columns);
    }

    /// <summary>
    /// Creates a new matrix filled with ones, with the specified number of rows and columns.
    /// </summary>
    /// <param name="rows">The number of rows in the matrix.</param>
    /// <param name="columns">The number of columns in the matrix.</param>
    /// <returns>A new matrix filled with ones.</returns>
    public static Matrix Ones(int rows, int columns)
    {
        // Create an instance of Array of floats using rows and columns and fill it with ones.
        Array array = Array.CreateInstance(typeof(float), rows, columns);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array.SetValue(1f, i, j);
            }
        }
        return new Matrix(array);
    }

    /// <summary>
    /// Creates a new matrix filled with random values between -0.5 and 0.5, with the specified number of rows and columns.
    /// </summary>
    /// <param name="rows">The number of rows in the matrix.</param>
    /// <param name="columns">The number of columns in the matrix.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>A new matrix filled with random values.</returns>
    public static Matrix Random(int rows, int columns, Random random)
    {
        // Create an instance of Array of floats using rows and columns and fill it with randoms.
        Array array = Array.CreateInstance(typeof(float), rows, columns);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array.SetValue(random.NextSingle() - 0.5f, i, j);
            }
        }
        return new Matrix(array);
    }

    #endregion

    #region Operations with scalar

    /// <summary>
    /// Adds a scalar value to each element of the matrix.
    /// </summary>
    /// <param name="scalar">The scalar value to add.</param>
    /// <returns>A new matrix with the scalar added to each element.</returns>
    public Matrix Add(float scalar)
    {
        (Array array, int rows, int columns) = GetCopyAsArray();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                // ((float[,])array)[i, j] = ((float[,])_array)[i, j] + scalar;
                array.SetValue((float)_array.GetValue(i, j)! + scalar, i, j);
            }
        }

        return new Matrix(array);
    }

    /// <summary>
    /// Multiplies each element of the matrix by a scalar value.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply.</param>
    /// <returns>A new matrix with each element multiplied by the scalar value.</returns>
    public Matrix Multiply(float scalar)
    {
        (Array array, int rows, int columns) = GetCopyAsArray();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array.SetValue((float)_array.GetValue(i, j)! * scalar, i, j);
            }
        }

        return new Matrix(array);
    }

    /// <summary>
    /// Raises each element of the matrix to the specified power.
    /// </summary>
    /// <param name="scalar">The power to raise each element to.</param>
    /// <returns>A new matrix with each element raised to the specified power.</returns>
    public Matrix Power(int scalar)
    {
        (Array array, int rows, int columns) = GetCopyAsArray();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array.SetValue(MathF.Pow((float)_array.GetValue(i, j)!, scalar), i, j);
            }
        }

        return new Matrix(array);
    }

    #endregion

    #region Operations with matrix

    /// <summary>
    /// Multiplies the current matrix with another matrix using the dot product.
    /// </summary>
    /// <param name="matrix">The matrix to multiply with.</param>
    /// <returns>A new matrix that is the result of the dot product multiplication.</returns>
    public Matrix MultiplyDot(Matrix matrix)
    {
        if (GetDimension(Dimension.Columns) != matrix.GetDimension(Dimension.Rows))
            throw new Exception(NumberOfColumnsMustBeEqualToNumberOfRowsMsg);

        int matrixColumns = matrix.Array.GetLength(1);

        int rows = _array.GetLength(0);
        int columns = _array.GetLength(1);

        Array array = Array.CreateInstance(typeof(float), rows, matrixColumns);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < matrixColumns; j++)
            {
                float sum = 0;
                for (int k = 0; k < columns; k++)
                {
                    sum += (float)_array.GetValue(i, k)! * (float)matrix.Array.GetValue(k, j)!;
                }
                array.SetValue(sum, i, j);
            }
        }

        return new Matrix(array);
    }

    /// <summary>
    /// Multiplies each element of the matrix with the corresponding element of another matrix.
    /// </summary>
    /// <param name="matrix">The matrix to multiply elementwise with.</param>
    /// <returns>A new matrix with each element multiplied elementwise.</returns>
    public Matrix MultiplyElementwise(Matrix matrix)
    {
        if (GetDimension(Dimension.Rows) != matrix.GetDimension(Dimension.Rows))
            throw new Exception(NumberOfRowsMustBeEqualToNumberOfRowsMsg);

        (Array array, int rows, int columns) = GetCopyAsArray();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array.SetValue((float)_array.GetValue(i, j)! * (float)matrix.Array.GetValue(i, j)!, i, j);
            }
        }

        return new Matrix(array);
    }

    /// <summary>
    /// Subtracts the elements of the specified matrix from the current matrix.
    /// </summary>
    /// <param name="matrix">The matrix to subtract.</param>
    /// <returns>A new matrix with the elements subtracted.</returns>
    public Matrix Subtract(Matrix matrix)
    {
        if (GetDimension(Dimension.Rows) != matrix.GetDimension(Dimension.Rows))
            throw new Exception(NumberOfRowsMustBeEqualToNumberOfRowsMsg);

        if (GetDimension(Dimension.Columns) != matrix.GetDimension(Dimension.Columns))
            throw new Exception(NumberOfColumnsMustBeEqualToNumberOfColumnsMsg);

        (Array array, int rows, int columns) = GetCopyAsArray();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array.SetValue((float)_array.GetValue(i, j)! - (float)matrix.Array.GetValue(i, j)!, i, j);
            }
        }

        return new Matrix(array);
    }

    #endregion

    #region Aggregations

    /// <summary>
    /// Calculates the mean of all elements in the matrix.
    /// </summary>
    /// <returns>The mean of all elements in the matrix.</returns>
    public float Mean() => Sum() / _array.Length;

    /// <summary>
    /// Calculates the sum of all elements in the matrix.
    /// </summary>
    /// <returns>The sum of all elements in the matrix.</returns>
    public float Sum()
    {
        // return _array.Cast<float>().Sum();
        // Sum over all elements.
        float sum = 0;
        foreach (object? item in _array)
        {
            sum += (float)item!;
        }
        return sum;
    }

    #endregion

    #region Slices and Rows

    /// <summary>
    /// Gets a row from the matrix.
    /// </summary>
    /// <param name="row">The index of the row to retrieve.</param>
    /// <returns>A new <see cref="Matrix"/> object representing the specified row.</returns>
    /// <remarks>
    /// The returned row is a new instance of the <see cref="Matrix"/> class and has the same number of columns as the original matrix.
    /// </remarks>
    public Matrix GetRow(int row)
    {
        int columns = _array.GetLength(1);

        // Create an array to store the row.
        float[,] newArray = new float[1, columns];
        for (int i = 0; i < columns; i++)
        {
            // Access each element in the specified row.
            newArray[0, i] = (float)_array.GetValue(row, i)!;
        }

        return new Matrix(newArray);
    }

    /// <summary>
    /// Sets the values of a specific row in the matrix.
    /// </summary>
    /// <param name="row">The index of the row to set.</param>
    /// <param name="matrix">The matrix containing the values to set.</param>
    /// <exception cref="Exception">Thrown when the number of columns in the specified matrix is not equal to the number of columns in the current matrix.</exception>
    public void SetRow(int row, Matrix matrix)
    {
        if (matrix.GetDimension(Dimension.Columns) != _array.GetLength(1))
            throw new Exception(NumberOfColumnsMustBeEqualToNumberOfColumnsMsg);

        for (int i = 0; i < _array.GetLength(1); i++)
        {
            _array.SetValue(matrix.Array.GetValue(0, i), row, i);
        }
    }

    /// <summary>
    /// Gets a submatrix containing the specified range of rows from the current matrix.
    /// </summary>
    /// <param name="range">The range of rows to retrieve.</param>
    /// <returns>A new <see cref="Matrix"/> object representing the submatrix.</returns>
    /// <remarks>
    /// The returned rows are a new instance of the <see cref="Matrix"/> class and have the same number of columns as the original matrix.
    /// </remarks>
    public Matrix GetRows(Range range)
    {
        (int offset, int length) = range.GetOffsetAndLength(_array.GetLength(0));

        Array newArray = Array.CreateInstance(typeof(float), length, _array.GetLength(1));

        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < _array.GetLength(1); j++)
            {
                newArray.SetValue(_array.GetValue(i + offset, j), i, j);
            }
        }

        return new Matrix(newArray);
    }

    #endregion

    #region Matrix operations

    /// <summary>
    /// Transposes the matrix by swapping its rows and columns.
    /// </summary>
    /// <returns>A new <see cref="Matrix"/> object representing the transposed matrix.</returns>
    public Matrix Transpose()
    {
        int rows = _array.GetLength(0);
        int columns = _array.GetLength(1);

        Array array = Array.CreateInstance(typeof(float), columns, rows);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array.SetValue(_array.GetValue(i, j), j, i);
            }
        }

        return new Matrix(array);
    }

    #endregion

    /// <summary>
    /// Gets the number of rows or columns in the matrix.
    /// </summary>
    /// <param name="dimension">The dimension to get the size of.</param>
    public int GetDimension(Dimension dimension) => _array.GetLength((int)dimension);

    private static (int Rows, int Columns) GetDimensions(Matrix inputMatrix) => (inputMatrix.GetDimension(Dimension.Rows), inputMatrix.GetDimension(Dimension.Columns));

    private (Array Array, int Rows, int Columns) GetCopyAsArray()
    {
        int rows = _array.GetLength(0);
        int columns = _array.GetLength(1);
        Array array = Array.CreateInstance(typeof(float), rows, columns);
        return (array, rows, columns);
    }

    // Some commented out code experimenting with different ways to implement the indexer, and slicing.

    //internal MyArray GetSlice(int dimension, int index)
    //{
    //    // if the dimension is 0 then return the column
    //    if (dimension == 0)
    //    {
    //        return new MyArray(_array.GetValue(0, index) as Array);
    //    }

    //    // if the dimension is 1 then return the row
    //    return new MyArray(_array.GetValue(index, 0) as Array);
    //}

    //internal MyArray this[params int[] index]
    //{
    //    get
    //    {
    //        float[] newArray = new float[_array.GetLength(1)]; // Create an array to store the second row
    //        for (int i = 0; i < _array.GetLength(1); i++)
    //        {
    //            newArray[i] = (float)_array.GetValue(index[0], i); // Access each element in the second row
    //        }

    //        // return a new MyArray instance with the specified index
    //        return new MyArray(newArray);
    //    }
    //    set
    //    {

    //        // set the value of the specified index
    //        _array.SetValue(value.Array, index);
    //    }
    //}

    //internal MyArray this[Range range]
    //{
    //    get
    //    {
    //        (int offset, int length) = range.GetOffsetAndLength(_array.GetLength(0));

    //        Array newArray = Array.CreateInstance(typeof(float), length, _array.GetLength(1));

    //        Array.Copy(_array, offset, newArray, 0, length);

    //        return new MyArray(newArray);
    //    }
    //    //set
    //    //{
    //    //    throw new NotImplementedException();
    //    //}
    //}
}