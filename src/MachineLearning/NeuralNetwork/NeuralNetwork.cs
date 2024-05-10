// Machine Learning Utils
// File name: NeuralNetwork.cs
// Code It Yourself with .NET, 2024

using MachineLearning.NeuralNetwork.Layers;
using MachineLearning.NeuralNetwork.Losses;

namespace MachineLearning.NeuralNetwork;

public class NeuralNetwork(List<Layer> layers, Loss lossFunction)
{
    private readonly List<Layer> _layers = layers;
    private Loss _lossFunction = lossFunction;
    private float _lastLoss;

    public Loss LossFunction => _lossFunction;

    public float LastLoss => _lastLoss;

    public Matrix Forward(Matrix batch)
    {
        Matrix input = batch;
        foreach (Layer layer in _layers)
        {
            input = layer.Forward(input);
        }
        return input;
    }

    public void Backward(Matrix lossGrad)
    {
        Matrix grad = lossGrad;
        foreach (Layer layer in _layers.Reverse<Layer>())
        {
            grad = layer.Backward(grad);
        }
    }

    public float TrainBatch(Matrix xBatch, Matrix yBatch)
    {
        Matrix predictions = Forward(xBatch);
        _lastLoss = _lossFunction.Forward(predictions, yBatch);
        Backward(_lossFunction.Backward());
        return _lastLoss;
    }

    public Matrix[] GetParams() => _layers.SelectMany(layer => layer.Params).ToArray();

    public Matrix[] GetParamGradients() => _layers.SelectMany(layer => layer.ParamGradients).ToArray();

    private NeuralNetwork? _checkpoint;

    public void SaveCheckpoint() => _checkpoint = Clone();

    public bool HasCheckpoint() => _checkpoint is not null;

    public void RestoreCheckpoint()
    {
        if (_checkpoint is null)
        {
            throw new Exception("No checkpoint to restore.");
        }
        _layers.Clear();
        _layers.AddRange(_checkpoint._layers.Select(l => l.Clone()));
        _lossFunction = _checkpoint._lossFunction.Clone();
    }

    /// <summary>
    /// Makes a deep copy of this neural network.
    /// </summary>
    /// <returns></returns>
    public NeuralNetwork Clone()
    {
        return new NeuralNetwork(
            _layers.Select(l => l.Clone())
                .ToList(),
            _lossFunction.Clone()
        );
    }
}
