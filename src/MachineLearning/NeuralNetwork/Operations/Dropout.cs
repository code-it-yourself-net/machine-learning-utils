// Machine Learning Utils
// File name: Dropout.cs
// Code It Yourself with .NET, 2024

using MachineLearning.NeuralNetwork.Exceptions;

namespace MachineLearning.NeuralNetwork.Operations;

public class Dropout(float keepProb = 0.8f) : Operation
{
    private Matrix? _mask;

    protected override Matrix CalcOutput(bool inference)
    {
        if (inference)
        {
            return Input.Multiply(keepProb);
        }
        else
        {
            Random random = new();
            _mask = Matrix.ZeroOnes(Input, keepProb, random);

            return Input.MultiplyElementwise(_mask);
        }
    }

    protected override Matrix CalcInputGradient(Matrix outputGradient) 
        => outputGradient.MultiplyElementwise(_mask ?? throw new NotYetCalculatedException());
}
