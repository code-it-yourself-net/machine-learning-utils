// Machine Learning Utils
// File name: ParamOperation.cs
// Code It Yourself with .NET, 2024

// This class is derived from content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

using MachineLearning.NeuralNetwork.Exceptions;

using static MachineLearning.MatrixUtils;

namespace MachineLearning.NeuralNetwork.Operations;

/// <summary>
/// An Operation with parameters.
/// </summary>
/// <param name="param">Parameter matrix.</param>
public abstract class ParamOperation(MatrixOld param) : Operation
{
    private MatrixOld? _paramGradient;

    public MatrixOld Param => param;

    public MatrixOld ParamGradient => _paramGradient ?? throw new NotYetCalculatedException();

    public override MatrixOld Backward(MatrixOld outputGrad)
    {
        MatrixOld inputGrad = base.Backward(outputGrad);
        _paramGradient = CalcParamGradient(outputGrad);
        EnsureSameShape(param, _paramGradient);

        return inputGrad;
    }

    protected abstract MatrixOld CalcParamGradient(MatrixOld outputGrad);

    protected override Operation CloneBase() 
    { 
        ParamOperation clone = (ParamOperation)base.CloneBase();
        clone._paramGradient = _paramGradient?.Clone();
        return clone;
    }

}
