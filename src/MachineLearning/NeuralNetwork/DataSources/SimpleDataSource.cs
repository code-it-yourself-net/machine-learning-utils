// Machine Learning Utils
// File name: SimpleDataSource.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.DataSources;

public class SimpleDataSource(MatrixOld xTrain, MatrixOld yTrain, MatrixOld? xTest, MatrixOld? yTest) : DataSource
{
    public override (MatrixOld xTrain, MatrixOld yTrain, MatrixOld? xTest, MatrixOld? yTest) GetData() => (xTrain, yTrain, xTest, yTest);
}
