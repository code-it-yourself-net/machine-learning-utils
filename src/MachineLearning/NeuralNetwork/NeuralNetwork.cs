﻿// Machine Learning Utils
// File name: NeuralNetwork.cs
// Code It Yourself with .NET, 2024

using MachineLearning.NeuralNetwork.Layers;
using MachineLearning.NeuralNetwork.Losses;

namespace MachineLearning.NeuralNetwork;

public class NeuralNetwork(List<Layer> layers, Loss lossFunction)
{
    private List<Layer> _layers = layers;
    private Loss _lossFunction = lossFunction;
    private float _lastLoss;

    public IReadOnlyList<Layer> Layers => _layers;

    public Loss LossFunction => _lossFunction;

    public float LastLoss => _lastLoss;

    public int ParameterCount => _layers
        .SelectMany(layer => layer.Params)
        .Sum(paramMatrix => paramMatrix.Array.Length);

    public MatrixOld Forward(MatrixOld batch)
    {
        MatrixOld input = batch;
        foreach (Layer layer in _layers)
        {
            input = layer.Forward(input);
        }
        return input;
    }

    public void Backward(MatrixOld lossGrad)
    {
        MatrixOld grad = lossGrad;
        foreach (Layer layer in _layers.Reverse<Layer>())
        {
            grad = layer.Backward(grad);
        }
    }

    public float TrainBatch(MatrixOld xBatch, MatrixOld yBatch)
    {
        MatrixOld predictions = Forward(xBatch);
        _lastLoss = _lossFunction.Forward(predictions, yBatch);
        Backward(_lossFunction.Backward());
        return _lastLoss;
    }

    public MatrixOld[] GetParams() => _layers.SelectMany(layer => layer.Params).ToArray();

    public MatrixOld[] GetParamGradients() => _layers.SelectMany(layer => layer.ParamGradients).ToArray();

    private NeuralNetwork? _checkpoint;

    public void SaveCheckpoint() => _checkpoint = Clone();

    public bool HasCheckpoint() => _checkpoint is not null;

    public void RestoreCheckpoint()
    {
        if (_checkpoint is null)
        {
            throw new Exception("No checkpoint to restore.");
        }
        // _checkpoint is already a deep copy so we can just copy its fields.
        _layers = _checkpoint._layers;
        _lossFunction = _checkpoint._lossFunction;
        _lastLoss = _checkpoint._lastLoss;
    }

    /// <summary>
    /// Makes a deep copy of this neural network.
    /// </summary>
    /// <returns></returns>
    public NeuralNetwork Clone()
    {
        NeuralNetwork clone = (NeuralNetwork)MemberwiseClone();
        clone._layers = _layers.Select(l => l.Clone()).ToList();
        clone._lossFunction = _lossFunction.Clone();
        return clone;
    }
}
