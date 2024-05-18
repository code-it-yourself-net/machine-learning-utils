﻿// Machine Learning Utils
// File name: Dropout.cs
// Code It Yourself with .NET, 2024

using MachineLearning.NeuralNetwork.Exceptions;

namespace MachineLearning.NeuralNetwork.Operations;

public class Dropout(float keepProb = 0.8f, SeededRandom? random = null) : Operation
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
            _mask = Matrix.ZeroOnes(Input, keepProb, random ?? new());

            return Input.MultiplyElementwise(_mask);
        }
    }

    protected override Matrix CalcInputGradient(Matrix outputGradient) 
        => outputGradient.MultiplyElementwise(_mask ?? throw new NotYetCalculatedException());

    public override string ToString() => $"Dropout (keepProb={keepProb}, seed={random?.Seed})";
}
