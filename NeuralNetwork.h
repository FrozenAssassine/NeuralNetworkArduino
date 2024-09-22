#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Layers.h"

class NeuralNetwork {
  public:
    BaseLayer** allLayer;
    int stackingIndex;
    int totalLayers;
    NeuralNetwork(int totalLayers);
    ~NeuralNetwork();
    void Train(float* inputs, float* desired, int totalItems, int epochs, float learningRate);
    NeuralNetwork& StackLayer(BaseLayer* layer);
    void Build();
};

#endif