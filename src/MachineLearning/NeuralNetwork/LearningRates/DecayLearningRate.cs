// Machine Learning Utils
// File name: DecayLearningRate.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.LearningRates;

public abstract class DecayLearningRate : LearningRate
{
    public DecayLearningRate(float initialLearningRate)
    {
        CurrentLearningRate = initialLearningRate;
    }

    protected float CurrentLearningRate { get; set; }

    public override float GetLearningRate() => CurrentLearningRate;
}
