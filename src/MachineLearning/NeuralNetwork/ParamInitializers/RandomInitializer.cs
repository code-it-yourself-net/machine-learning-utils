// Machine Learning Utils
// File name: RandomInitializer.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.ParamInitializers;

public class RandomInitializer : ParamInitializer
{
    private readonly Random _random;
    private readonly int? _seed;

    public RandomInitializer(int? seed = null)
    {
        if (seed.HasValue)
            _random = new Random(seed.Value);
        else
            _random = new Random();
        _seed = seed;
    }

    protected Random Random => _random;

    protected int? Seed => _seed;

    internal override MatrixOld InitBiases(int neurons) 
        => MatrixOld.Random(1, neurons, _random);

    internal override MatrixOld InitWeights(int inputColumns, int neurons) 
        => MatrixOld.Random(inputColumns, neurons, _random);

    public override string ToString() => $"RandomInitializer (seed={_seed})";
}
