// Machine Learning Utils
// File name: RangeInitializer.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.ParamInitializers;

public class RangeInitializer(float from, float to) : ParamInitializer
{
    internal override Matrix InitBiases(int neurons) 
        => Matrix.Zeros(1, neurons);

    internal override Matrix InitWeights(int inputColumns, int neurons) 
        => Matrix.Range(inputColumns, neurons, from, to);

    public override string ToString() => $"RangeInitializer (from={from}, to={to})";
}
