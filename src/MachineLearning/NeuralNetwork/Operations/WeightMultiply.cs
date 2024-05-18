// Machine Learning Utils
// File name: WeightMultiply.cs
// Code It Yourself with .NET, 2024

// This class is derived from content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

namespace MachineLearning.NeuralNetwork.Operations;

/// <summary>
/// Weight multiplication operation for a neural network.
/// </summary>
/// <param name="weight">Weight matrix.</param>
public class WeightMultiply(MatrixOld weight) : ParamOperation(weight)
{
    protected override MatrixOld Output()
        => Input.MultiplyDot(Param);

    protected override MatrixOld InputGrad(MatrixOld outputGrad)
        => outputGrad.MultiplyDot(Param.Transpose());

    protected override MatrixOld CalcParamGradient(MatrixOld outputGrad)
        => Input.Transpose().MultiplyDot(outputGrad);
}
