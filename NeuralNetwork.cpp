#include "NeuralNetwork.h"
#include "Arduino.h"

NeuralNetwork::NeuralNetwork(int totalLayers) {
  this->allLayer = new BaseLayer*[totalLayers];
  this->totalLayers = totalLayers;
  this->stackingIndex = 0;
}

NeuralNetwork::~NeuralNetwork() {
  delete[] this->allLayer;
}

NeuralNetwork& NeuralNetwork::StackLayer(BaseLayer* layer) {
  if (this->stackingIndex >= this->totalLayers) {
    Serial.println("Cannot stack any more layers. Check your total layer count.");
    return *this;
  }
  this->allLayer[this->stackingIndex++] = layer;
  return *this;
}

void NeuralNetwork::Build() {
  for (int i = 0; i < this->totalLayers; i++) {
    if (i == 0) {  // first layer (input)
      allLayer[i]->InitLayer(allLayer[i]->Size, nullptr, allLayer[i + 1]);
    } else if (i == this->totalLayers - 1) {  // output layer
      allLayer[i]->InitLayer(allLayer[i]->Size, allLayer[i - 1], nullptr);
    } else {
      allLayer[i]->InitLayer(allLayer[i]->Size, allLayer[i - 1], allLayer[i + 1]);
    }
  }
}

void NeuralNetwork::Train(float* inputs, float* desired, int totalItems, int epochs, float learningRate) {
  for (int epoch = 0; epoch < epochs; epoch++) {
    for (int i = 0; i < totalItems; i++) {
      //TODO: hardcoded index is not good!
      this->allLayer[0]->NeuronValues[0] = inputs[i * 2 + 0];
      this->allLayer[0]->NeuronValues[1] = inputs[i * 2 + 1];

      for (int j = 0; j < this->totalLayers; j++) {
        this->allLayer[j]->FeedForward();
      }

      for (int j = this->totalLayers - 1; j >= 0; j--) {
        this->allLayer[j]->Train(&desired[i], learningRate);
      }
    }

    if (epoch % 100 == 0) {
      Serial.print("EPOCH ");
      Serial.println(epoch);
    }
  }
}