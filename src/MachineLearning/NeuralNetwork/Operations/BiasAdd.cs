// Machine Learning Utils
// File name: BiasAdd.cs
// Code It Yourself with .NET, 2024

// This class is derived from content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

namespace MachineLearning.NeuralNetwork.Operations;

/// <summary>
/// Computes bias addition.
/// </summary>
/// <param name="bias">Bias matrix.</param>
public class BiasAdd(Matrix bias) : ParamOperation(ValidateBiasMatrix(bias))
{
    protected override Matrix Output() => Input.AddRow(Param);

    protected override Matrix InputGrad(Matrix outputGrad) => Matrix.Ones(Input).MultiplyElementwise(outputGrad);

    protected override Matrix CalcParamGradient(Matrix outputGrad)
    {
        Matrix paramGrad = Matrix.Ones(Param).MultiplyElementwise(outputGrad);
        // return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])
        return paramGrad.SumBy(Dimension.Rows); // Reshape - ?
    }

    private static Matrix ValidateBiasMatrix(Matrix bias)
    {
        if (bias.GetDimension(Dimension.Rows) != 1)
            throw new ArgumentException("Bias matrix must have one row.");

        return bias;
    }
}
