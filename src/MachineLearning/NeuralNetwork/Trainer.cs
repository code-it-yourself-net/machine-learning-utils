// Machine Learning Utils
// File name: Trainer.cs
// Code It Yourself with .NET, 2024

using MachineLearning.NeuralNetwork.DataSources;
using MachineLearning.NeuralNetwork.Optimizers;

using static MachineLearning.MatrixUtils;

namespace MachineLearning.NeuralNetwork;

public class Trainer(NeuralNetwork neuralNetwork, Optimizer optimizer)
{
    private float _bestLoss = float.MaxValue;

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
        DataSource dataSource,
        Func<NeuralNetwork, Matrix, Matrix, float>? evalFunction = null,
        int epochs = 100,
        int evalEveryEpochs = 10,
        int batchSize = 32,
        bool printOnlyEvalEpochs = false,
        bool restart = true)
    {
        (Matrix xTrain, Matrix yTrain, Matrix? xTest, Matrix? yTest) = dataSource.GetData();

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            bool evaluationEpoch = epoch % evalEveryEpochs == 0;
            bool eval = xTest is not null && yTest is not null && evaluationEpoch;

            if(evaluationEpoch || !printOnlyEvalEpochs)
                Console.WriteLine($"Epoch {epoch}/{epochs}...");

            if (eval)
            {
                neuralNetwork.SaveCheckpoint();
            }

            (xTrain, yTrain) = PermuteData(xTrain, yTrain, new Random(123)); // TODO: random

            float? trainLoss = null;
            int step = 0;
            int allSteps = (int)Math.Ceiling(xTrain.GetDimension(Dimension.Rows) / (float)batchSize);
            foreach ((Matrix xBatch, Matrix yBatch) in GenerateBatches(xTrain, yTrain, batchSize))
            {
                step++;
                Console.Write($"Step {step}/{allSteps}...\r");

                trainLoss = (trainLoss ?? 0) + neuralNetwork.TrainBatch(xBatch, yBatch);
                optimizer.Step(neuralNetwork);
            }

            if (trainLoss is not null && evaluationEpoch)
            {
                Console.WriteLine($"Train loss (average): {trainLoss.Value / allSteps}");
            }

            if (eval)
            {
                Matrix testPredictions = neuralNetwork.Forward(xTest!);
                float loss = neuralNetwork.LossFunction.Forward(testPredictions, yTest!);
                Console.WriteLine($"Test loss: {loss}");

                if (evalFunction is not null)
                {
                    float evalValue = evalFunction(neuralNetwork, xTest!, yTest!);
                    Console.WriteLine($"Eval: {evalValue:P2}");
                }

                if (loss < _bestLoss)
                {
                    _bestLoss = loss;
                }
                else
                {
                    if (neuralNetwork.HasCheckpoint())
                        neuralNetwork.RestoreCheckpoint();
                    Console.WriteLine("Early stopping.");
                    break;
                }

            }
        }
    }
}
