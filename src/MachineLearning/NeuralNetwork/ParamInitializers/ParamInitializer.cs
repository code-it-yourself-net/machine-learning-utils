// Machine Learning Utils
// File name: ParamInitializer.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.ParamInitializers;

public abstract class ParamInitializer
{
    internal abstract Matrix InitBiases(int neurons);
    internal abstract Matrix InitWeights(int inputColumns, int neurons);
}
