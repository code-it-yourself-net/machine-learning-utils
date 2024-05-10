// Machine Learning Utils
// File name: Optimizer.cs
// Code It Yourself with .NET, 2024

// This class is derived from content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

namespace MachineLearning.NeuralNetwork.Optimizers;

/// <summary>
/// Base class for a neural network optimizer.
/// </summary>
public abstract class Optimizer
{
    private readonly float _learningRate;

    protected float LearningRate => _learningRate;

    public Optimizer(float learningRate = 0.01f)
    {
        _learningRate = learningRate;
    }

    public abstract void Step(NeuralNetwork neuralNetwork);
}
