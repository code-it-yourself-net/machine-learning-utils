// Machine Learning Utils
// File name: TypedVsUntypedVsFlat.cs
// Code It Yourself with .NET, 2024

using BenchmarkDotNet.Attributes;

using MachineLearning;

namespace MatrixBenchmark;

public class TypedVsUntypedVsFlat
{
    Matrix _matrix1Untyped = null!;
    Matrix _matrix2Untyped = null!;
    TypedMatrix _matrix1Typed = null!;
    TypedMatrix _matrix2Typed = null!;

    // [Params(100, 1000)]
    [Params(1000)]
    public int N;

    [GlobalSetup]
    public void Setup()
    {
        float[,] matrix1 = new float[790, 89];
        float[,] matrix2 = new float[89, 10];

        // fill in matrix1 and matrix2 with random float numbers
        Random random = new(909);
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

        _matrix1Untyped = new(matrix1);
        _matrix2Untyped = new(matrix2);

        _matrix1Typed = new(matrix1);
        _matrix2Typed = new(matrix2);
    }

    /*
    [Benchmark]
    public void UntypedMatrixMultiplication()
    {
        Matrix result = _matrix1Untyped.MultiplyDot(_matrix2Untyped);
    }

    [Benchmark]
    public void TypedMatrixMultiplication()
    {
        TypedMatrix result = _matrix1Typed.MultiplyDot(_matrix2Typed);
    }

    [Benchmark]
    public void TypedMatrixMultiplicationWithMatrixArray()
    {
        TypedMatrix result = _matrix1Typed.MultiplyDotWithMatrixArray(_matrix2Typed);
    }

    [Benchmark]
    public void UntypedSigmoid()
    {
        Matrix result = _matrix1Untyped.Sigmoid();
    }

    [Benchmark]
    public void TypedSigmoid()
    {
        TypedMatrix result = _matrix1Typed.Sigmoid();
    }
    */
    [Benchmark]
    public void Std()
    {
        float max = _matrix1Typed.Std();
    }

    [Benchmark]
    public void StdTyped()
    {
        float max = _matrix1Typed.StdTyped();
    }

    /*
    [Benchmark]
    public void MaxLoopTyped()
    {
        float max = _matrix1Typed.MaxLoopTyped();
    }
    */
}
