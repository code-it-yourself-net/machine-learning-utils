// Machine Learning Utils
// File name: SoftmaxCrossEntropyLoss.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.Losses;

public class SoftmaxCrossEntropyLoss(float eps = 1e-7f) : Loss
{
    protected override float CalculateLoss()
    {
        // Calculate the probabilities.
        Matrix softmaxPrediction = Prediction.Softmax();

        // Clip the probabilities to avoid log(0).
        softmaxPrediction.ClipInPlace(eps, 1 - eps);

        Matrix negativeTarget = Target.Multiply(-1f);
        Matrix softmaxCrossEntropyLoss = negativeTarget.MultiplyElementwise(softmaxPrediction.Log())
            .Subtract(
                negativeTarget.Add(1f).MultiplyElementwise(softmaxPrediction.Multiply(-1f).Add(1f).Log())
            );
        int batchSize = Prediction.GetDimension(Dimension.Rows);
        return softmaxCrossEntropyLoss.Sum() / batchSize;
    }

    protected override Matrix CalculateLossGradient()
    {
        Matrix softmaxPrediction = Prediction.Softmax();
        int batchSize = Prediction.GetDimension(Dimension.Rows);
        return softmaxPrediction.Subtract(Target).Divide(batchSize);
    }
}
