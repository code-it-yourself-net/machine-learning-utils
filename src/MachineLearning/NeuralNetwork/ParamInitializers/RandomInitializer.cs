﻿// Machine Learning Utils
// File name: RandomInitializer.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.ParamInitializers;

public class RandomInitializer : ParamInitializer
{
    private readonly Random _random;

    public RandomInitializer(int? seed = null)
    {
        if (seed.HasValue)
            _random = new Random(seed.Value);
        else
            _random = new Random();
    }

    internal override Matrix InitBiases(int neurons) 
        => Matrix.Random(1, neurons, _random);

    internal override Matrix InitWeights(int inputColumns, int neurons) 
        => Matrix.Random(inputColumns, neurons, _random);
}
