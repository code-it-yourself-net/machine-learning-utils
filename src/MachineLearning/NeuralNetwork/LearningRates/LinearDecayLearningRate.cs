// Machine Learning Utils
// File name: LinearDecayLearningRate.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.LearningRates;

public class LinearDecayLearningRate : DecayLearningRate
{
    private readonly float _initialLearningRate;
    private readonly float _finalLearningRate;

    public LinearDecayLearningRate(float initialLearningRate, float finalLearningRate): base(initialLearningRate)
    {
        _initialLearningRate = initialLearningRate;
        _finalLearningRate = finalLearningRate;
    }

    public override void Update(int epoch, int epochs) => CurrentLearningRate = _initialLearningRate - (_initialLearningRate - _finalLearningRate) * epoch / epochs;

    public override string ToString() => $"LinearDecayLearningRate (initialLearningRate={_initialLearningRate}, finalLearningRate={_finalLearningRate})";
}
