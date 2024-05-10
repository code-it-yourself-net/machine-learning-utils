// Machine Learning Utils
// File name: ParamOperation.cs
// Code It Yourself with .NET, 2024

// This class is derived from content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

using static MachineLearning.MatrixUtils;

namespace MachineLearning.NeuralNetwork.Operations;

/// <summary>
/// An Operation with parameters.
/// </summary>
/// <param name="param">Parameter matrix.</param>
public abstract class ParamOperation(Matrix param) : Operation
{
    private Matrix? _paramGradient;

    public Matrix Param => param;

    public Matrix ParamGradient => _paramGradient ?? throw new Exception();

    public override Matrix Backward(Matrix outputGrad)
    {
        Matrix inputGrad = base.Backward(outputGrad);
        _paramGradient = CalcParamGradient(outputGrad);
        EnsureSameShape(param, _paramGradient);

        return inputGrad;
    }

    protected abstract Matrix CalcParamGradient(Matrix outputGrad);
}
