// Machine Learning Utils
// File name: GlorotInitializer.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.ParamInitializers;

public class GlorotInitializer(int? seed = null) : RandomInitializer(seed)
{
    internal override MatrixOld InitWeights(int inputColumns, int neurons)
    {
        float stdDev = (float)Math.Sqrt(2.0 / (inputColumns + neurons));
        return MatrixOld.RandomNormal(inputColumns, neurons, Random, 0, stdDev);
    }

    internal override MatrixOld InitBiases(int neurons) => MatrixOld.Zeros(1, neurons);

    public override string ToString() => $"GlorotInitializer (seed={Seed})";
}