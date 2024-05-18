// Machine Learning Utils
// File name: MatrixUtils.cs
// Code It Yourself with .NET, 2024

using System.Diagnostics;

namespace MachineLearning;

public static class MatrixUtils
{
    public static void EnsureSameShape(MatrixOld? matrix1, MatrixOld? matrix2)
    {
        if (matrix1 is null || matrix2 is null)
            throw new ArgumentException("Matrix is null.");

        if (!matrix1.HasSameShape(matrix2))
            throw new Exception("Matrices must have the same shape.");
    }

    public static (MatrixOld xPermuted, MatrixOld yPermuted) PermuteData(MatrixOld x, MatrixOld y, Random random)
    {
        Debug.Assert(x.GetDimension(Dimension.Rows) == y.GetDimension(Dimension.Rows));

        int[] indices = [.. Enumerable.Range(0, x.GetDimension(Dimension.Rows)).OrderBy(i => random.Next())];

        MatrixOld xPermuted = MatrixOld.Zeros(x);
        MatrixOld yPermuted = MatrixOld.Zeros(y);

        for (int i = 0; i < x.GetDimension(Dimension.Rows); i++)
        {
            //xPermuted[i] = x[indices[i]];
            //yPermuted[i] = y[indices[i]];
            xPermuted.SetRow(i, x.GetRow(indices[i]));
            yPermuted.SetRow(i, y.GetRow(indices[i]));
        }

        return (xPermuted, yPermuted);
    }
}
