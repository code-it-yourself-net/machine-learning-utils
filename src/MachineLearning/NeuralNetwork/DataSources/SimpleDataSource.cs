// Machine Learning Utils
// File name: SimpleDataSource.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.DataSources;

public class SimpleDataSource(Matrix xTrain, Matrix yTrain, Matrix? xTest, Matrix? yTest) : DataSource
{
    public override (Matrix xTrain, Matrix yTrain, Matrix? xTest, Matrix? yTest) GetData() => (xTrain, yTrain, xTest, yTest);
}
