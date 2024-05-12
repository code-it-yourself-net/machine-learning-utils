// Machine Learning Utils
// File name: Tanh.cs
// Code It Yourself with .NET, 2024

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning.NeuralNetwork.Operations;

public class Tanh : Operation
{
    protected override Matrix Output() => Input.Tanh();

    protected override Matrix InputGrad(Matrix outputGrad)
    {
        // tanh_backward = 1 - self.output * self.output
        Matrix tanhBackward = Matrix.Ones(Output()).Subtract(Output().MultiplyElementwise(Output()));
        // input_grad = tanh_backward * output_grad
        return outputGrad.MultiplyElementwise(tanhBackward);
    }
}
