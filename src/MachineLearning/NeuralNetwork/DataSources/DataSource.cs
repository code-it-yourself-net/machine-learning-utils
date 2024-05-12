// Machine Learning Utils
// File name: DataSource.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.DataSources;

public abstract class DataSource
{
    public abstract (Matrix xTrain, Matrix yTrain, Matrix? xTest, Matrix? yTest) GetData();
}
