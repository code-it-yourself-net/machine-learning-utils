// Machine Learning Utils
// File name: RangeInitializer.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.ParamInitializers;

public class RangeInitializer(float from, float to) : ParamInitializer
{
    internal override MatrixOld InitBiases(int neurons) 
        => MatrixOld.Zeros(1, neurons);

    internal override MatrixOld InitWeights(int inputColumns, int neurons) 
        => MatrixOld.Range(inputColumns, neurons, from, to);

    public override string ToString() => $"RangeInitializer (from={from}, to={to})";
}
