// Machine Learning Utils
// File name: NormallyDistributedRandomInitializer.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.ParamInitializers;

public class NormallyDistributedRandomInitializer(float mean = 0, float stdDev = 1, SeededRandom? random = null) : RandomInitializer(random)
{
    internal override Matrix InitBiases(int neurons) => Matrix.RandomNormal(1, neurons, Random, mean, stdDev);

    internal override Matrix InitWeights(int inputColumns, int neurons) => Matrix.RandomNormal(inputColumns, neurons, Random, mean, stdDev);

    public override string ToString() => $"NormallyDistributedRandomInitializer (seed={Seed}, mean={mean}, stdDev={stdDev})";
}
