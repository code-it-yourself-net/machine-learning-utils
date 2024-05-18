// Machine Learning Utils
// File name: DataSource.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.DataSources;

public abstract class DataSource
{
    public abstract (MatrixOld xTrain, MatrixOld yTrain, MatrixOld? xTest, MatrixOld? yTest) GetData();
}
