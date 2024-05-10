// Machine Learning Utils
// File name: Trainer.cs
// Code It Yourself with .NET, 2024

using static MachineLearning.MatrixUtils;

using MachineLearning.NeuralNetwork.Optimizers;

namespace MachineLearning.NeuralNetwork;

public class Trainer
{
    private NeuralNetwork _neuralNetwork;
    private readonly Optimizer _optimizer;
    private float _bestLoss;

    public Trainer(NeuralNetwork neuralNetwork, Optimizer optimizer)
    {
        _neuralNetwork = neuralNetwork;
        _optimizer = optimizer;
        _bestLoss = 1e9f;
    }

    public static IEnumerable<(Matrix xBatch, Matrix yBatch)> GenerateBatches(Matrix x, Matrix y, int batchSize = 32)
    {
        int trainRows = x.GetDimension(Dimension.Rows);
        if (trainRows != y.GetDimension(Dimension.Rows))
        {
            throw new ArgumentException("Number of samples in x and y do not match.");
        }

        for (int batchStart = 0; batchStart < trainRows; batchStart += batchSize)
        {
            int effectiveBatchSize = Math.Min(batchSize, trainRows - batchStart);
            int batchEnd = effectiveBatchSize + batchStart;
            Matrix xBatch = x.GetRows(batchStart..batchEnd);
            Matrix yBatch = y.GetRows(batchStart..batchEnd);
            yield return (xBatch, yBatch);
        }
    }

    public void Fit(
        Matrix xTrain,
        Matrix yTrain,
        Matrix? xTest,
        Matrix? yTest,
        int epochs = 100,
        int evalEveryEpochs = 10,
        int batchSize = 32,
        bool restart = true)
    {
        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            if (xTest is not null && yTest is not null && (epoch % evalEveryEpochs == 0))
            {
                _neuralNetwork.SaveCheckpoint();
            }

            (xTrain, yTrain) = PermuteData(xTrain, yTrain, new Random()); // TODO: random

            foreach ((Matrix xBatch, Matrix yBatch) in GenerateBatches(xTrain, yTrain, batchSize))
            {
                float loss = _neuralNetwork.TrainBatch(xBatch, yBatch);
                if(epoch % evalEveryEpochs == 0)
                {
                    Console.WriteLine($"Epoch {epoch}, Loss: {loss}");
                }
                _optimizer.Step(_neuralNetwork);
            }

            if (xTest is not null && yTest is not null && ((epoch + 1) % evalEveryEpochs == 0))
            {
                Matrix testPredictions = _neuralNetwork.Forward(xTest);
                float loss = _neuralNetwork.LossFunction.Forward(testPredictions, yTest);
                Console.WriteLine($"Epoch {epoch + 1}, Loss: {loss}");

                if (loss < _bestLoss)
                {
                    _bestLoss = loss;
                }
                else
                {
                    Console.WriteLine("Early stopping.");
                    break;
                }

            }
        }
    }
}
