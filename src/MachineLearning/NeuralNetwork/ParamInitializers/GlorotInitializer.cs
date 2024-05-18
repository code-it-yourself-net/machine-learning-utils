// Machine Learning Utils
// File name: GlorotInitializer.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.ParamInitializers;

public class GlorotInitializer(SeededRandom? random = null) : RandomInitializer(random)
{
    internal override Matrix InitWeights(int inputColumns, int neurons)
    {
        float stdDev = (float)Math.Sqrt(2.0 / (inputColumns + neurons));
        return Matrix.RandomNormal(inputColumns, neurons, Random, 0, stdDev);
    }

    internal override Matrix InitBiases(int neurons) => Matrix.Zeros(1, neurons);

    public override string ToString() => $"GlorotInitializer (seed={Seed})";
}