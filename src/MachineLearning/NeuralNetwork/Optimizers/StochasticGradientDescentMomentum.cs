// Machine Learning Utils
// File name: StochasticGradientDescentMomentum.cs
// Code It Yourself with .NET, 2024

using MachineLearning.NeuralNetwork.LearningRates;

namespace MachineLearning.NeuralNetwork.Optimizers;

public class StochasticGradientDescentMomentum(LearningRate learningRate, float momentum) : Optimizer(learningRate)
{
    private MatrixOld[]? _velocities;

    public override void Step(NeuralNetwork neuralNetwork)
    {
        MatrixOld[] @params = neuralNetwork.GetParams();
        MatrixOld[] paramGrads = neuralNetwork.GetParamGradients();

        if (@params.Length != paramGrads.Length)
        {
            throw new ArgumentException("Number of parameters and gradients do not match.");
        }

        if (_velocities == null)
        {
            _velocities = new MatrixOld[@params.Length];
            for (int i = 0; i < @params.Length; i++)
            {
                _velocities[i] = MatrixOld.Zeros(@params[i]);
            }
        }

        // Iterate through both lists in parallel
        for (int i = 0; i < @params.Length; i++)
        {
            MatrixOld param = @params[i];
            MatrixOld paramGrad = paramGrads[i];
            MatrixOld velocity = _velocities[i];

            // Update the velocity
            velocity.MultiplyInPlace(momentum);
            MatrixOld deltaParamGrad = paramGrad.Multiply(LearningRate.GetLearningRate());
            velocity.AddInPlace(deltaParamGrad);

            // Update the parameter
            param.SubtractInPlace(velocity);
        }
    }

    public override string ToString() => $"StochasticGradientDescentMomentum (learningRate={LearningRate}, momentum={momentum})";
}