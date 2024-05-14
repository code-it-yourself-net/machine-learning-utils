// Machine Learning Utils
// File name: Dense.cs
// Code It Yourself with .NET, 2024

// This class is derived from content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

using MachineLearning.NeuralNetwork.Operations;
using MachineLearning.NeuralNetwork.ParamInitializers;

namespace MachineLearning.NeuralNetwork.Layers;

public class DenseLayer(int neurons, Operation activation, ParamInitializer paramInitializer) : Layer(neurons)
{
    protected override void SetupLayer(Matrix input)
    {
        Matrix weights = paramInitializer.InitWeights(input.GetDimension(Dimension.Columns), Neurons);
        Matrix biases = paramInitializer.InitBiases(Neurons);

        // Uncomment for tests:
        //Matrix weights = new(new float[2, 1] { { -1f }, { 1f } });
        //Matrix biases = new(new float[1, 1] { { 0f } });

        Params.AddRange([
            weights,
            biases
        ]);

        Operations.AddRange([
            new WeightMultiply(weights),
            new BiasAdd(biases),
            activation
        ]);
    }

    #region Clone

    protected override Layer CloneBase()
    {
        DenseLayer clone = (DenseLayer)base.CloneBase();
        return clone;
    }

    #endregion

    public override string ToString() => $"DenseLayer (neurons={Neurons}, activation={activation}, paramInitializer={paramInitializer})";
}
