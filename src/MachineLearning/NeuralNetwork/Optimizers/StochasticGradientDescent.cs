// Machine Learning Utils
// File name: StochasticGradientDescent.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.Optimizers;

public class StochasticGradientDescent(float learningRate) : Optimizer(learningRate)
{
    public override void Step(NeuralNetwork neuralNetwork)
    {
        Matrix[] @params = neuralNetwork.GetParams();
        Matrix[] paramGrads = neuralNetwork.GetParamGradients();

        if (@params.Length != paramGrads.Length)
        {
            throw new ArgumentException("Number of parameters and gradients do not match.");
        }

        // Iterate through both lists in parallel
        for (int i = 0; i < @params.Length; i++)
        {
            Matrix param = @params[i];
            Matrix paramGrad = paramGrads[i];

            // Update the parameter
            Matrix deltaParamGrad = paramGrad.Multiply(LearningRate);
            param.SubtractInPlace(deltaParamGrad);
        }
    }

    public override string ToString() => $"Stochastic Gradient Descent (lr={LearningRate})";
}
