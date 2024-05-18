// Machine Learning Utils
// File name: Dropout.cs
// Code It Yourself with .NET, 2024

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning.NeuralNetwork.Operations;

public class Dropout: Operation
{
    private readonly float _keepProb;
    private readonly Matrix _mask;

    public Dropout(float keepProb = 0.8f)
    {
        _keepProb = keepProb;
        _mask = new Matrix(0, 0);
    }

    protected override Matrix CalcOutput() => throw new NotImplementedException();

    protected override Matrix CalcInputGradient(Matrix outputGrad) => throw new NotImplementedException();

    /*
    protected override Matrix Output() { }

    public override Matrix Forward(Matrix input, bool isTraining)
    {
        if (!isTraining)
        {
            return input;
        }

        _mask.Resize(input.Rows, input.Columns);
        _mask.FillRandom(0, 1);
        _mask.MapInPlace(x => x < _keepProb ? 1 : 0);

        return input * _mask / _keepProb;
    }

    public override Matrix Backward(Matrix outputGradient)
    {
        return outputGradient * _mask / _keepProb;
    }
    */
}
