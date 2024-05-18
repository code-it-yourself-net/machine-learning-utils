// Machine Learning Utils
// File name: Operation.cs
// Code It Yourself with .NET, 2024

// This class is derived from content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

using MachineLearning.NeuralNetwork.Exceptions;

using static MachineLearning.MatrixUtils;

namespace MachineLearning.NeuralNetwork.Operations;

/// <summary>
/// Base class for an "operation" in a neural network.
/// </summary>
public abstract class Operation
{
    private MatrixOld? _input;
    private MatrixOld? _inputGrad;
    private MatrixOld? _output;

    protected MatrixOld Input => _input ?? throw new NotYetCalculatedException();

    public virtual MatrixOld Forward(MatrixOld input)
    {
        _input = input;
        _output = Output();
        return _output;
    }

    public virtual MatrixOld Backward(MatrixOld outputGrad)
    {
        EnsureSameShape(_output, outputGrad);
        _inputGrad = InputGrad(outputGrad);

        EnsureSameShape(_input, _inputGrad);
        return _inputGrad;
    }

    /// <summary>
    /// Computes output.
    /// </summary>
    protected abstract MatrixOld Output();

    /// <summary>
    /// Computes input gradient.
    /// </summary>
    protected abstract MatrixOld InputGrad(MatrixOld outputGrad);

    #region Clone

    protected virtual Operation CloneBase()
    {
        Operation clone =(Operation)MemberwiseClone();
        clone._input = _input?.Clone();
        clone._inputGrad = _inputGrad?.Clone();
        clone._output = _output?.Clone();
        return clone;
    }

    public Operation Clone() => CloneBase();

    #endregion
}
