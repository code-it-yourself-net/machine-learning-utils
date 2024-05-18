// Machine Learning Utils
// File name: PermutableDataSource.cs
// Code It Yourself with .NET, 2024

using static MachineLearning.MatrixUtils;

namespace MachineLearning.NeuralNetwork.DataSources;

public abstract class PermutableDataSource : DataSource
{
    private readonly float _testFraction;
    private readonly SeededRandom? _random;

    public PermutableDataSource(float testFraction, SeededRandom? random = null)
    {
        if (testFraction < 0 || testFraction >= 1)
            throw new ArgumentOutOfRangeException(nameof(testFraction), "Test fraction must be greater than or equal to 0 and less than 1.");

        _testFraction = testFraction;
        _random = random;
    }

    public override (Matrix xTrain, Matrix yTrain, Matrix? xTest, Matrix? yTest) GetData()
    {
        Matrix xTrain, yTrain;
        Matrix? xTest, yTest;

        (Matrix x, Matrix y) = GetAllData();

        int allRows = x.GetDimension(Dimension.Rows);
        int testRows = (int)Math.Round(allRows * _testFraction);

        if (testRows > 0)
        {
            (Matrix xPermuted, Matrix yPermuted) = PermuteData(x, y, _random ?? new Random());
            xTest = xPermuted.GetRows(0..testRows);
            yTest = yPermuted.GetRows(0..testRows);
            xTrain = xPermuted.GetRows(testRows..);
            yTrain = yPermuted.GetRows(testRows..);
        }
        else
        {
            xTest = null;
            yTest = null;
            xTrain = x;
            yTrain = y;
        }

        return (xTrain, yTrain, xTest, yTest);
    }

    public abstract (Matrix x, Matrix y) GetAllData();
}
