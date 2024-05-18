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
    protected override MatrixOld Output() => Input.Tanh();

    protected override MatrixOld InputGrad(MatrixOld outputGrad)
    {
        // tanh_backward = 1 - self.output * self.output
        MatrixOld tanhBackward = MatrixOld.Ones(Output()).Subtract(Output().MultiplyElementwise(Output()));
        // input_grad = tanh_backward * output_grad
        return outputGrad.MultiplyElementwise(tanhBackward);
    }

    public override string ToString() => "Tanh";
}
