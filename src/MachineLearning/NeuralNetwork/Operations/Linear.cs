// Machine Learning Utils
// File name: Linear.cs
// Code It Yourself with .NET, 2024

// This class is derived from content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

namespace MachineLearning.NeuralNetwork.Operations;

/// <summary>
/// "Identity" activation function
/// </summary>
public class Linear : Operation
{
    protected override Matrix Output() => Input;

    protected override Matrix InputGrad(Matrix outputGrad) => outputGrad;
}
