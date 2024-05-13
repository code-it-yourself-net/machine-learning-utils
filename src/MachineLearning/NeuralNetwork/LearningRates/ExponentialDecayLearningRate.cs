// Machine Learning Utils
// File name: ExponentialDecayLearningRate.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.LearningRates;

public class ExponentialDecayLearningRate : DecayLearningRate
{
    private readonly float _initialLearningRate;
    private readonly float _finalLearningRate;

    public ExponentialDecayLearningRate(float initialLearningRate, float finalLearningRate): base(initialLearningRate)
    {
        _initialLearningRate = initialLearningRate;
        _finalLearningRate = finalLearningRate;
    }

    public override void Update(int epoch, int epochs) => CurrentLearningRate = _initialLearningRate * (float)Math.Pow(_finalLearningRate / _initialLearningRate, (float)epoch / epochs);

    public override string ToString() => $"ExponentialDecayLearningRate (initialLearningRate={_initialLearningRate}, finalLearningRate={_finalLearningRate})";
}
