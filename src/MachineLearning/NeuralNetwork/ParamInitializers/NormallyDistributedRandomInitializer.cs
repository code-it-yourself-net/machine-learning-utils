// Machine Learning Utils
// File name: NormallyDistributedRandomInitializer.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.ParamInitializers;

public class NormallyDistributedRandomInitializer(float mean = 0, float stdDev = 1, int? seed = null) : RandomInitializer(seed)
{
    internal override MatrixOld InitBiases(int neurons) => MatrixOld.RandomNormal(1, neurons, Random, mean, stdDev);

    internal override MatrixOld InitWeights(int inputColumns, int neurons) => MatrixOld.RandomNormal(inputColumns, neurons, Random, mean, stdDev);

    public override string ToString() => $"NormallyDistributedRandomInitializer (seed={Seed}, mean={mean}, stdDev={stdDev})";
}
