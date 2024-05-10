// Machine Learning Utils
// File name: Loss.cs
// Code It Yourself with .NET, 2024

// This class is derived from content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

using static MachineLearning.MatrixUtils;

namespace MachineLearning.NeuralNetwork.Losses;

/// <summary>
/// The "loss" of a neural network.
/// </summary>
public abstract class Loss
{
    private Matrix? _predictions;
    private Matrix? _target;

    protected Matrix Predictions => _predictions ?? throw new Exception();
    protected Matrix Target => _target ?? throw new Exception();

    /// <summary>
    /// Computes the actual loss value
    /// </summary>
    public float Forward(Matrix predictions, Matrix target)
    {
        EnsureSameShape(predictions, target);
        _predictions = predictions;
        _target = target;

        return CalculateLoss();
    }

    /// <summary>
    /// Computes gradient of the loss value with respect to the input to the loss function.
    /// </summary>
    public Matrix Backward()
    {
        Matrix lossGradient = CalculateLossGradient();
        EnsureSameShape(_predictions, lossGradient);
        return lossGradient;
    }

    protected abstract float CalculateLoss();

    protected abstract Matrix CalculateLossGradient();

    internal virtual Loss Clone()
    {
        // create a deep copy of this

        return (Loss)MemberwiseClone();
    }
}
