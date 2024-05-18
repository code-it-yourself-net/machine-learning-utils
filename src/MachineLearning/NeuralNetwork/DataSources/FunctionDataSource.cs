// Machine Learning Utils
// File name: FunctionDataSource.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.DataSources;

public class FunctionDataSource(float[,] arguments, Func<float[], float> function, float testFraction, int? seed = null) : PermutableDataSource(testFraction, seed)
{

    public override (MatrixOld x, MatrixOld y) GetAllData()
    {
        int argRows = arguments.GetLength(0);
        int argColumns = arguments.GetLength(1);

        float[,] yData = new float[argRows, 1];

        for (int row = 0; row < argRows; row++)
        {
            float[] rowData = new float[argColumns];
            for (int column = 0; column < argColumns; column++)
            {
                rowData[column] = arguments[row, column];
            }
            yData[row, 0] = function(rowData);
        }

        MatrixOld x = new(arguments);
        MatrixOld y = new(yData);

        return (x, y);
    }
}
