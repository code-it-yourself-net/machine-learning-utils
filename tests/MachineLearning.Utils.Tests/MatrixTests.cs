// Machine Learning Utils
// File name: UnitTest1.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.Utils.Tests;

[TestClass]
public class MatrixTests
{
    [TestMethod]
    public void ZerosTest()
    {
        var matrix = Matrix.Zeros(2, 3);
        Assert.AreEqual(2, matrix.GetDimension(Dimension.Rows));
        Assert.AreEqual(3, matrix.GetDimension(Dimension.Columns));
        Assert.AreEqual(0f, matrix.Array.GetValue(0, 0));
        Assert.AreEqual(0f, matrix.Array.GetValue(1, 2));
    }

    [TestMethod]
    public void OnesTest()
    {
        var matrix = Matrix.Ones(3, 2);
        Assert.AreEqual(3, matrix.GetDimension(Dimension.Rows));
        Assert.AreEqual(2, matrix.GetDimension(Dimension.Columns));
        Assert.AreEqual(1f, matrix.Array.GetValue(0, 0));
        Assert.AreEqual(1f, matrix.Array.GetValue(2, 1));
    }

    [TestMethod]
    public void RandomTest()
    {
        var random = new Random();
        var matrix = Matrix.Random(2, 2, random);
        Assert.AreEqual(2, matrix.GetDimension(Dimension.Rows));
        Assert.AreEqual(2, matrix.GetDimension(Dimension.Columns));
        Assert.IsTrue((float)matrix.Array.GetValue(0, 0)! >= -0.5f && (float)matrix.Array.GetValue(0, 0)! <= 0.5f);
        Assert.IsTrue((float)matrix.Array.GetValue(1, 1)! >= -0.5f && (float)matrix.Array.GetValue(1, 1)! <= 0.5f);
    }

    [TestMethod]
    public void AddTest()
    {
        var matrix = new Matrix(new float[,] { { 1, 2 }, { 3, 4 } });
        var result = matrix.Add(2);
        Assert.AreEqual(3f, result.Array.GetValue(0, 0));
        Assert.AreEqual(6f, result.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void MultiplyTest()
    {
        var matrix = new Matrix(new float[,] { { 1, 2 }, { 3, 4 } });
        var result = matrix.Multiply(2);
        Assert.AreEqual(2f, result.Array.GetValue(0, 0));
        Assert.AreEqual(8f, result.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void PowerTest()
    {
        var matrix = new Matrix(new float[,] { { 1, 2 }, { 3, 4 } });
        var result = matrix.Power(2);
        Assert.AreEqual(1f, result.Array.GetValue(0, 0));
        Assert.AreEqual(16f, result.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void MultiplyDotTest()
    {
        var matrix1 = new Matrix(new float[,] { { 1, 2 }, { 3, 4 } });
        var matrix2 = new Matrix(new float[,] { { 5, 6 }, { 7, 8 } });
        var result = matrix1.MultiplyDot(matrix2);
        Assert.AreEqual(19f, result.Array.GetValue(0, 0));
        Assert.AreEqual(22f, result.Array.GetValue(0, 1));
        Assert.AreEqual(43f, result.Array.GetValue(1, 0));
        Assert.AreEqual(50f, result.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void MultiplyElementwiseTest()
    {
        var matrix1 = new Matrix(new float[,] { { 1, 2 }, { 3, 4 } });
        var matrix2 = new Matrix(new float[,] { { 5, 6 }, { 7, 8 } });
        var result = matrix1.MultiplyElementwise(matrix2);
        Assert.AreEqual(5f, result.Array.GetValue(0, 0));
        Assert.AreEqual(12f, result.Array.GetValue(0, 1));
        Assert.AreEqual(21f, result.Array.GetValue(1, 0));
        Assert.AreEqual(32f, result.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void SubtractTest()
    {
        var matrix1 = new Matrix(new float[,] { { 1, 2 }, { 3, 4 } });
        var matrix2 = new Matrix(new float[,] { { 5, 6 }, { 7, 8 } });
        var result = matrix1.Subtract(matrix2);
        Assert.AreEqual(-4f, result.Array.GetValue(0, 0));
        Assert.AreEqual(-4f, result.Array.GetValue(0, 1));
        Assert.AreEqual(-4f, result.Array.GetValue(1, 0));
        Assert.AreEqual(-4f, result.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void MeanTest()
    {
        var matrix = new Matrix(new float[,] { { 1, 2 }, { 3, 4 } });
        var result = matrix.Mean();
        Assert.AreEqual(2.5f, result);
    }

    [TestMethod]
    public void SumTest()
    {
        var matrix = new Matrix(new float[,] { { 1, 2 }, { 3, 4 } });
        var result = matrix.Sum();
        Assert.AreEqual(10, result);
    }

    [TestMethod]
    public void GetRowTest()
    {
        var matrix = new Matrix(new float[,] { { 1, 2 }, { 3, 4 } });
        var row = matrix.GetRow(1);
        Assert.AreEqual(1, row.GetDimension(Dimension.Rows));
        Assert.AreEqual(2, row.GetDimension(Dimension.Columns));
        Assert.AreEqual(3f, row.Array.GetValue(0, 0));
        Assert.AreEqual(4f, row.Array.GetValue(0, 1));
    }

    [TestMethod]
    public void SetRowTest()
    {
        var matrix = new Matrix(new float[,] { { 1, 2 }, { 3, 4 } });
        var row = new Matrix(new float[,] { { 5, 6 } });
        matrix.SetRow(1, row);
        Assert.AreEqual(1f, matrix.Array.GetValue(0, 0));
        Assert.AreEqual(2f, matrix.Array.GetValue(0, 1));
        Assert.AreEqual(5f, matrix.Array.GetValue(1, 0));
        Assert.AreEqual(6f, matrix.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void GetRowsTest()
    {
        var matrix = new Matrix(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var submatrix = matrix.GetRows(1..3);
        Assert.AreEqual(2, submatrix.GetDimension(Dimension.Rows));
        Assert.AreEqual(2, submatrix.GetDimension(Dimension.Columns));
        Assert.AreEqual(3f, submatrix.Array.GetValue(0, 0));
        Assert.AreEqual(4f, submatrix.Array.GetValue(0, 1));
        Assert.AreEqual(5f, submatrix.Array.GetValue(1, 0));
        Assert.AreEqual(6f, submatrix.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void TransposeTest()
    {
        var matrix = new Matrix(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var transposed = matrix.Transpose();
        Assert.AreEqual(2, transposed.GetDimension(Dimension.Rows));
        Assert.AreEqual(3, transposed.GetDimension(Dimension.Columns));
        Assert.AreEqual(1f, transposed.Array.GetValue(0, 0));
        Assert.AreEqual(2f, transposed.Array.GetValue(1, 0));
        Assert.AreEqual(3f, transposed.Array.GetValue(0, 1));
        Assert.AreEqual(4f, transposed.Array.GetValue(1, 1));
        Assert.AreEqual(5f, transposed.Array.GetValue(0, 2));
        Assert.AreEqual(6f, transposed.Array.GetValue(1, 2));
    }
}
