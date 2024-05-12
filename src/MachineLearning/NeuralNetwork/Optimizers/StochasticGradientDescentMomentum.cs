// Machine Learning Utils
// File name: StochasticGradientDescentMomentum.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.Optimizers;

public class StochasticGradientDescentMomentum(float learningRate, float momentum) : Optimizer(learningRate)
{
    private Matrix[]? _velocities;

    public override void Step(NeuralNetwork neuralNetwork)
    {
        Matrix[] @params = neuralNetwork.GetParams();
        Matrix[] paramGrads = neuralNetwork.GetParamGradients();

        if (@params.Length != paramGrads.Length)
        {
            throw new ArgumentException("Number of parameters and gradients do not match.");
        }

        if (_velocities == null)
        {
            _velocities = new Matrix[@params.Length];
            for (int i = 0; i < @params.Length; i++)
            {
                _velocities[i] = Matrix.Zeros(@params[i]);
            }
        }

        // Iterate through both lists in parallel
        for (int i = 0; i < @params.Length; i++)
        {
            Matrix param = @params[i];
            Matrix paramGrad = paramGrads[i];
            Matrix velocity = _velocities[i];

            // Update the velocity
            velocity.MultiplyInPlace(momentum);
            Matrix deltaParamGrad = paramGrad.Multiply(LearningRate);
            velocity.AddInPlace(deltaParamGrad);

            //Matrix deltaParamGrad = paramGrad.Multiply(LearningRate);
            //Matrix deltaVelocity = velocity.Multiply(momentum).Add(deltaParamGrad);
            //velocity.Assign(deltaVelocity);

            // Update the parameter
            param.SubtractInPlace(velocity);
        }
    }

    public override string ToString() => $"Stochastic Gradient Descent with Momentum (lr={LearningRate}, momentum={momentum})";
}