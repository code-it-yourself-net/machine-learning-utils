// Machine Learning Utils
// File name: TypedVsUntypedVsFlat.cs
// Code It Yourself with .NET, 2024

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using BenchmarkDotNet.Attributes;

using MachineLearning;

namespace MatrixBenchmark;

public class TypedVsUntypedVsFlat
{
    [GlobalSetup]
    public void Setup()
    {
        float[,] matrix1 = new float[790, 89];
        float[,] matrix2 = new float[89, 10];

        // fill in matrix1 and matrix2 with random float numbers
        Random random = new Random(909);
        for (int i = 0; i < matrix1.GetLength(0); i++)
        {
            for (int j = 0; j < matrix1.GetLength(1); j++)
            {
                matrix1[i, j] = (float)random.NextDouble();
            }
        }

        for (int i = 0; i < matrix2.GetLength(0); i++)
        {
            for (int j = 0; j < matrix2.GetLength(1); j++)
            {
                matrix2[i, j] = (float)random.NextDouble();
            }
        }

        Matrix matrix1Untyped = new Matrix(matrix1);
        Matrix matrix2Untyped = new Matrix(matrix2);

    }
}
