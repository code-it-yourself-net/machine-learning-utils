// Machine Learning Utils
// File name: MatrixTests.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.Tests;

[TestClass]
public class MatrixTests
{
    [TestMethod]
    public void ZerosTest()
    {
        Matrix matrix = Matrix.Zeros(2, 3);
        Assert.AreEqual(2, matrix.GetDimension(Dimension.Rows));
        Assert.AreEqual(3, matrix.GetDimension(Dimension.Columns));
        Assert.AreEqual(0f, matrix.Array.GetValue(0, 0));
        Assert.AreEqual(0f, matrix.Array.GetValue(1, 2));
    }

    [TestMethod]
    public void OnesTest()
    {
        Matrix matrix = Matrix.Ones(3, 2);
        Assert.AreEqual(3, matrix.GetDimension(Dimension.Rows));
        Assert.AreEqual(2, matrix.GetDimension(Dimension.Columns));
        Assert.AreEqual(1f, matrix.Array.GetValue(0, 0));
        Assert.AreEqual(1f, matrix.Array.GetValue(2, 1));
    }

    [TestMethod]
    public void RandomTest()
    {
        Random random = new();
        Matrix matrix = Matrix.Random(2, 2, random);
        Assert.AreEqual(2, matrix.GetDimension(Dimension.Rows));
        Assert.AreEqual(2, matrix.GetDimension(Dimension.Columns));
        Assert.IsTrue((float)matrix.Array.GetValue(0, 0)! >= -0.5f && (float)matrix.Array.GetValue(0, 0)! <= 0.5f);
        Assert.IsTrue((float)matrix.Array.GetValue(1, 1)! >= -0.5f && (float)matrix.Array.GetValue(1, 1)! <= 0.5f);
    }

    [TestMethod]
    public void AddTest()
    {
        Matrix matrix = new(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix result = matrix.Add(2);
        Assert.AreEqual(3f, result.Array.GetValue(0, 0));
        Assert.AreEqual(6f, result.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void AddRowTest()
    {
        Matrix matrix = new(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix row = new(new float[,] { { 5, 6 } });
        Matrix result = matrix.AddRow(row);
        Assert.AreEqual(6f, result.Array.GetValue(0, 0));
        Assert.AreEqual(8f, result.Array.GetValue(0, 1));
        Assert.AreEqual(8f, result.Array.GetValue(1, 0));
        Assert.AreEqual(10f, result.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void MultiplyTest()
    {
        Matrix matrix = new(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix result = matrix.Multiply(2);
        Assert.AreEqual(2f, result.Array.GetValue(0, 0));
        Assert.AreEqual(8f, result.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void PowerTest()
    {
        Matrix matrix = new(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix result = matrix.Power(2);
        Assert.AreEqual(1f, result.Array.GetValue(0, 0));
        Assert.AreEqual(16f, result.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void MultiplyDotTest()
    {
        Matrix matrix1 = new(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix matrix2 = new(new float[,] { { 5, 6 }, { 7, 8 } });
        Matrix result = matrix1.MultiplyDot(matrix2);
        Assert.AreEqual(19f, result.Array.GetValue(0, 0));
        Assert.AreEqual(22f, result.Array.GetValue(0, 1));
        Assert.AreEqual(43f, result.Array.GetValue(1, 0));
        Assert.AreEqual(50f, result.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void MultiplyElementwiseTest1()
    {
        Matrix matrix1 = new(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix matrix2 = new(new float[,] { { 5, 6 }, { 7, 8 } });
        Matrix result = matrix1.MultiplyElementwise(matrix2);
        Assert.AreEqual(5f, result.Array.GetValue(0, 0));
        Assert.AreEqual(12f, result.Array.GetValue(0, 1));
        Assert.AreEqual(21f, result.Array.GetValue(1, 0));
        Assert.AreEqual(32f, result.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void MultiplyRowElementwiseTest1()
    {
        Matrix matrix1 = new(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix matrix2 = new(new float[,] { { 5, 6 } });
        Matrix result = matrix1.MultiplyElementwise(matrix2);
        Assert.AreEqual(5f, result.Array.GetValue(0, 0));
        Assert.AreEqual(12f, result.Array.GetValue(0, 1));
        Assert.AreEqual(15f, result.Array.GetValue(1, 0));
        Assert.AreEqual(24f, result.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void MultiplyRowElementwiseTest2()
    {
        Matrix matrix1 = new(new float[,] { { 5, 6 } });
        Matrix matrix2 = new(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix result = matrix1.MultiplyElementwise(matrix2);
        Assert.AreEqual(5f, result.Array.GetValue(0, 0));
        Assert.AreEqual(12f, result.Array.GetValue(0, 1));
        Assert.AreEqual(15f, result.Array.GetValue(1, 0));
        Assert.AreEqual(24f, result.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void MultiplyColumnElementwiseTest1()
    {
        Matrix matrix1 = new(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix matrix2 = new(new float[,] { { 5 }, { 6 } });
        Matrix result = matrix1.MultiplyElementwise(matrix2);
        Assert.AreEqual(5f, result.Array.GetValue(0, 0));
        Assert.AreEqual(10f, result.Array.GetValue(0, 1));
        Assert.AreEqual(18f, result.Array.GetValue(1, 0));
        Assert.AreEqual(24f, result.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void MultiplyColumnElementwiseTest2()
    {
        Matrix matrix1 = new(new float[,] { { 5 }, { 6 } });
        Matrix matrix2 = new(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix result = matrix1.MultiplyElementwise(matrix2);
        Assert.AreEqual(5f, result.Array.GetValue(0, 0));
        Assert.AreEqual(10f, result.Array.GetValue(0, 1));
        Assert.AreEqual(18f, result.Array.GetValue(1, 0));
        Assert.AreEqual(24f, result.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void SubtractTest()
    {
        Matrix matrix1 = new(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix matrix2 = new(new float[,] { { 5, 6 }, { 7, 8 } });
        Matrix result = matrix1.Subtract(matrix2);
        Assert.AreEqual(-4f, result.Array.GetValue(0, 0));
        Assert.AreEqual(-4f, result.Array.GetValue(0, 1));
        Assert.AreEqual(-4f, result.Array.GetValue(1, 0));
        Assert.AreEqual(-4f, result.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void MeanTest()
    {
        Matrix matrix = new(new float[,] { { 1, 2 }, { 3, 4 } });
        float result = matrix.Mean();
        Assert.AreEqual(2.5f, result);
    }

    [TestMethod]
    public void SumTest()
    {
        Matrix matrix = new(new float[,] { { 1, 2 }, { 3, 4 } });
        float result = matrix.Sum();
        Assert.AreEqual(10, result);
    }

    [TestMethod]
    public void SumByTest()
    {
        Matrix matrix = new(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix result = matrix.SumBy(Dimension.Rows);
        Assert.AreEqual(1, result.GetDimension(Dimension.Rows));
        Assert.AreEqual(4f, result.Array.GetValue(0, 0));
        Assert.AreEqual(6f, result.Array.GetValue(0, 1));
    }

    [TestMethod]
    public void GetRowTest()
    {
        Matrix matrix = new(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix row = matrix.GetRow(1);
        Assert.AreEqual(1, row.GetDimension(Dimension.Rows));
        Assert.AreEqual(2, row.GetDimension(Dimension.Columns));
        Assert.AreEqual(3f, row.Array.GetValue(0, 0));
        Assert.AreEqual(4f, row.Array.GetValue(0, 1));
    }

    [TestMethod]
    public void GetColumnTest()
    {
        Matrix matrix = new(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix row = matrix.GetColumn(1);
        Assert.AreEqual(2, row.GetDimension(Dimension.Rows));
        Assert.AreEqual(1, row.GetDimension(Dimension.Columns));
        Assert.AreEqual(2f, row.Array.GetValue(0, 0));
        Assert.AreEqual(4f, row.Array.GetValue(1, 0));
    }

    [TestMethod]
    public void SetRowTest()
    {
        Matrix matrix = new(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix row = new(new float[,] { { 5, 6 } });
        matrix.SetRow(1, row);
        Assert.AreEqual(1f, matrix.Array.GetValue(0, 0));
        Assert.AreEqual(2f, matrix.Array.GetValue(0, 1));
        Assert.AreEqual(5f, matrix.Array.GetValue(1, 0));
        Assert.AreEqual(6f, matrix.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void GetRowsTest()
    {
        Matrix matrix = new(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        Matrix submatrix = matrix.GetRows(1..3);
        Assert.AreEqual(2, submatrix.GetDimension(Dimension.Rows));
        Assert.AreEqual(2, submatrix.GetDimension(Dimension.Columns));
        Assert.AreEqual(3f, submatrix.Array.GetValue(0, 0));
        Assert.AreEqual(4f, submatrix.Array.GetValue(0, 1));
        Assert.AreEqual(5f, submatrix.Array.GetValue(1, 0));
        Assert.AreEqual(6f, submatrix.Array.GetValue(1, 1));
    }

    [TestMethod]
    public void GetColumnsTest()
    {
        Matrix matrix = new(new float[,] { { 1, 2, 3 }, { 3, 4, 5 }, { 5, 6, 7 } });
        Matrix submatrix = matrix.GetColumns(1..3);
        Assert.AreEqual(3, submatrix.GetDimension(Dimension.Rows));
        Assert.AreEqual(2, submatrix.GetDimension(Dimension.Columns));
        Assert.AreEqual(2f, submatrix.Array.GetValue(0, 0));
        Assert.AreEqual(3f, submatrix.Array.GetValue(0, 1));
        Assert.AreEqual(4f, submatrix.Array.GetValue(1, 0));
        Assert.AreEqual(5f, submatrix.Array.GetValue(1, 1));
        Assert.AreEqual(6f, submatrix.Array.GetValue(2, 0));
        Assert.AreEqual(7f, submatrix.Array.GetValue(2, 1));
    }

    [TestMethod]
    public void TransposeTest()
    {
        Matrix matrix = new(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        Matrix transposed = matrix.Transpose();
        Assert.AreEqual(2, transposed.GetDimension(Dimension.Rows));
        Assert.AreEqual(3, transposed.GetDimension(Dimension.Columns));
        Assert.AreEqual(1f, transposed.Array.GetValue(0, 0));
        Assert.AreEqual(2f, transposed.Array.GetValue(1, 0));
        Assert.AreEqual(3f, transposed.Array.GetValue(0, 1));
        Assert.AreEqual(4f, transposed.Array.GetValue(1, 1));
        Assert.AreEqual(5f, transposed.Array.GetValue(0, 2));
        Assert.AreEqual(6f, transposed.Array.GetValue(1, 2));
    }

    [TestMethod]
    public void StdsAreEqual()
    {
        Matrix matrix = new(new float[,] { { 1, 2 }, { 3, 4 }, { 3, 4 }, { 5, 6 } });
        TypedMatrix typedMatrix = new(new float[,] { { 1, 2 }, { 3, 4 }, { 3, 4 }, { 5, 6 } });
        float std1 = matrix.Std();
        float std2 = typedMatrix.Std();
        Assert.AreEqual(std1, std2);
    }

    [TestMethod]
    public void SoftmaxAreEqual()
    {
        Matrix matrix = new(new float[,] { { 1, 2 }, { 3, 4 }, { 3, 4 }, { 5, 6 } });
        TypedMatrix typedMatrix = new(new float[,] { { 1, 2 }, { 3, 4 }, { 3, 4 }, { 5, 6 } });
        Matrix matrix1 = matrix.Softmax();
        TypedMatrix matrix2 = typedMatrix.Softmax();
        //Assert.AreEqual(matrix1, matrix2);

        // assert that the two matrices are equal
        Assert.AreEqual(matrix1.GetDimension(Dimension.Rows), matrix2.GetDimension(Dimension.Rows));
        Assert.AreEqual(matrix1.GetDimension(Dimension.Columns), matrix2.GetDimension(Dimension.Columns));
        for (int i = 0; i < matrix1.GetDimension(Dimension.Rows); i++)
        {
            for (int j = 0; j < matrix1.GetDimension(Dimension.Columns); j++)
            {
                Assert.AreEqual(matrix1.Array.GetValue(i, j), matrix2.Array.GetValue(i, j));
            }
        }
    }
}
