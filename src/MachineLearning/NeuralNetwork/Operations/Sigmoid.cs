// Machine Learning Utils
// File name: Sigmoid.cs
// Code It Yourself with .NET, 2024

// This class is derived from content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

namespace MachineLearning.NeuralNetwork.Operations;

/// <summary>
/// Sigmoid activation function.
/// </summary>
public class Sigmoid : Operation
{
    protected override MatrixOld Output() => Input.Sigmoid();

    protected override MatrixOld InputGrad(MatrixOld outputGrad)
    {
        // sigmoid_backward = self.output * (1.0 - self.output)
        MatrixOld sigmoidBackward = Output().MultiplyElementwise(MatrixOld.Ones(Output()).Subtract(Output()));
        // input_grad = sigmoid_backward * output_grad
        return outputGrad.MultiplyElementwise(sigmoidBackward);
    }

    public override string ToString() => "Sigmoid";
}
