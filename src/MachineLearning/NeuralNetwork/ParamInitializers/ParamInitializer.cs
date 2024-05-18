// Machine Learning Utils
// File name: ParamInitializer.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.ParamInitializers;

public abstract class ParamInitializer
{
    internal abstract MatrixOld InitBiases(int neurons);
    internal abstract MatrixOld InitWeights(int inputColumns, int neurons);
}
