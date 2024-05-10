// Machine Learning Utils
// File name: Operation.cs
// Code It Yourself with .NET, 2024

// This class is derived from content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

using static MachineLearning.MatrixUtils;

namespace MachineLearning.NeuralNetwork.Operations;

/// <summary>
/// Base class for an "operation" in a neural network.
/// </summary>
public abstract class Operation
{
    private Matrix? _input;
    private Matrix? _inputGrad;
    private Matrix? _output;

    protected Matrix Input => _input ?? throw new Exception();

    public virtual Matrix Forward(Matrix input)
    {
        _input = input;
        _output = Output();
        return _output;
    }

    public virtual Matrix Backward(Matrix outputGrad)
    {
        EnsureSameShape(_output, outputGrad);
        _inputGrad = InputGrad(outputGrad);

        EnsureSameShape(_input, _inputGrad);
        return _inputGrad;
    }

    /// <summary>
    /// Computes output.
    /// </summary>
    protected abstract Matrix Output();

    /// <summary>
    /// Computes input gradient.
    /// </summary>
    protected abstract Matrix InputGrad(Matrix outputGrad);
}
