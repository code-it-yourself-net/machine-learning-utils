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
    private Matrix? _input;
    private Matrix? _inputGradient;
    private Matrix? _output;

    protected Matrix Input => _input ?? throw new NotYetCalculatedException();
    protected Matrix Output => _output ?? throw new NotYetCalculatedException();

    public virtual Matrix Forward(Matrix input, bool inference)
    {
        _input = input;
        _output = CalcOutput(inference);
        return _output;
    }

    public virtual Matrix Backward(Matrix outputGradient)
    {
        EnsureSameShape(_output, outputGradient);
        _inputGradient = CalcInputGradient(outputGradient);

        EnsureSameShape(_input, _inputGradient);
        return _inputGradient;
    }

    /// <summary>
    /// Computes output.
    /// </summary>
    protected abstract Matrix CalcOutput(bool inference);

    /// <summary>
    /// Computes input gradient.
    /// </summary>
    protected abstract Matrix CalcInputGradient(Matrix outputGradient);

    #region Clone

    protected virtual Operation CloneBase()
    {
        Operation clone =(Operation)MemberwiseClone();
        clone._input = _input?.Clone();
        clone._inputGradient = _inputGradient?.Clone();
        clone._output = _output?.Clone();
        return clone;
    }

    public Operation Clone() => CloneBase();

    #endregion
}
