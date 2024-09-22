#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Layers.h"

class NeuralNetwork {
private:
  void TrainSingle(float* input, float* desired, int inputLength, float learningRate);
public:
  BaseLayer** allLayer;
  int stackingIndex;
  int totalLayers;
  NeuralNetwork(int totalLayers);
  ~NeuralNetwork();
  void Train(float* inputs, float* desired, int totalItems, int inputItemCount, int epochs, float learningRate);
  NeuralNetwork& StackLayer(BaseLayer* layer);
  void Build();
  float* Predict(float* inputs, int inputLength);
};

#endif