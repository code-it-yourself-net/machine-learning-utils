// Machine Learning Utils
// File name: ExponentialDecayLearningRate.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.LearningRates;

public class ExponentialDecayLearningRate(float initialLearningRate, float finalLearningRate) : DecayLearningRate(initialLearningRate)
{
    private readonly float _initialLearningRate = initialLearningRate;
    private readonly float _finalLearningRate = finalLearningRate;

    public override void Update(int epoch, int epochs) => CurrentLearningRate = _initialLearningRate * (float)Math.Pow(_finalLearningRate / _initialLearningRate, (float)epoch / epochs);

    public override string ToString() => $"ExponentialDecayLearningRate (initialLearningRate={_initialLearningRate}, finalLearningRate={_finalLearningRate})";
}
