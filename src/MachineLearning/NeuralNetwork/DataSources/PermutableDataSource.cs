// Machine Learning Utils
// File name: PermutableDataSource.cs
// Code It Yourself with .NET, 2024

using static MachineLearning.MatrixUtils;

namespace MachineLearning.NeuralNetwork.DataSources;

public abstract class PermutableDataSource : DataSource
{
    private readonly float _testFraction;
    private readonly int? _seed;

    public PermutableDataSource(float testFraction, int? seed = null)
    {
        if (testFraction < 0 || testFraction >= 1)
            throw new ArgumentOutOfRangeException(nameof(testFraction), "Test fraction must be greater than or equal to 0 and less than 1.");

        _testFraction = testFraction;
        _seed = seed;
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
            Random random;
            if (_seed.HasValue)
                random = new Random(_seed.Value);
            else
                random = new Random();

            (Matrix xPermuted, Matrix yPermuted) = PermuteData(x, y, random);
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
