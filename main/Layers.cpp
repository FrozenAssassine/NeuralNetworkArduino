#include "Layers.h"
#include "Arduino.h"

float Sigmoid(float x) {
  return 1.0f / (1.0f + exp(-x));
}

float SigmoidDeriv(float x) {
  return x * (1 - x);
}

void FillRandom(float* array, int size) {
  for (int i = 0; i < size; i++) {
    float rand = (random(0, 32768) / 32768.0f) * 2 - 1;
    array[i] = rand;
  }
}

///BASE LAYER
BaseLayer::BaseLayer()
  : Biases(nullptr), NeuronValues(nullptr), Errors(nullptr), Weights(nullptr),
    Size(0), PreviousLayer(nullptr), NextLayer(nullptr) {}
BaseLayer::~BaseLayer() {
  delete[] Biases;
  delete[] NeuronValues;
  delete[] Errors;
  delete[] Weights;
}

///DENSE LAYER
DenseLayer::DenseLayer(int size)
  : BaseLayer() {
  this->Size = size;
}
DenseLayer::~DenseLayer() {}
void DenseLayer::InitLayer(int size, BaseLayer* previous, BaseLayer* next) {
  this->Size = size;
  this->Biases = new float[size];
  this->NeuronValues = new float[size];
  this->Errors = new float[size];
  this->Weights = new float[size * previous->Size];
  this->PreviousLayer = previous;
  this->NextLayer = next;

  FillRandom(this->Biases, size);
  FillRandom(this->Weights, size * previous->Size);
}
void DenseLayer::FeedForward() {
  for (int idx = 0; idx < this->Size; idx++) {
    float sum = 0.0f;
    int index = idx * this->PreviousLayer->Size;
    for (int j = 0; j < this->PreviousLayer->Size; j++) {
      sum += this->PreviousLayer->NeuronValues[j] * this->Weights[index + j];
    }
    this->NeuronValues[idx] = Sigmoid(sum + this->Biases[idx]);
  }
}
void DenseLayer::Train(const float* desiredValues, float learningRate) {
  for (int idx = 0; idx < this->Size; idx++) {
    float err = 0.0f;
    int index = idx * this->PreviousLayer->Size;

    for (int j = 0; j < this->NextLayer->Size; j++) {
      err += (this->NextLayer->Errors[j] * this->NextLayer->Weights[j * this->Size + idx]);
    }
    float error = err * SigmoidDeriv(this->NeuronValues[idx]);
    this->Errors[idx] = error;

    error *= learningRate;

    for (int j = 0; j < this->PreviousLayer->Size; j++) {
      this->Weights[index + j] += error * this->PreviousLayer->NeuronValues[j];
    }

    this->Biases[idx] += error;
  }
}

//INPUT LAYER
InputLayer::InputLayer(int size)
  : BaseLayer() {
  this->Size = size;
}
InputLayer::~InputLayer() {}
void InputLayer::InitLayer(int size, BaseLayer* previous, BaseLayer* next) {
  this->Size = size;
  this->NeuronValues = new float[size];
  this->PreviousLayer = previous;
  this->NextLayer = next;
}
void InputLayer::FeedForward() {}
void InputLayer::Train(const float* desiredValues, float learningRate) {}


//OUTPUT LAYER
OutputLayer::OutputLayer(int size)
  : BaseLayer() {
  this->Size = size;
}
OutputLayer::~OutputLayer() {}
void OutputLayer::InitLayer(int size, BaseLayer* previous, BaseLayer* next) {
  this->Size = size;
  this->Biases = new float[size];
  this->NeuronValues = new float[size];
  this->Errors = new float[size];
  this->Weights = new float[size * previous->Size];
  this->PreviousLayer = previous;
  this->NextLayer = next;

  FillRandom(this->Biases, size);
  FillRandom(this->Weights, size * previous->Size);
}

void OutputLayer::FeedForward() {
  for (int idx = 0; idx < this->Size; idx++) {
    float sum = 0.0f;
    int weightIndex = idx * this->PreviousLayer->Size;
    for (int j = 0; j < this->PreviousLayer->Size; j++) {
      sum += this->PreviousLayer->NeuronValues[j] * this->Weights[weightIndex + j];
    }
    this->NeuronValues[idx] = Sigmoid(sum + this->Biases[idx]);
  }
}
void OutputLayer::Train(const float* desiredValues, float learningRate) {
  for (int idx = 0; idx < this->Size; idx++) {
    this->Errors[idx] = desiredValues[idx] - this->NeuronValues[idx];
  }

  for (int idx = 0; idx < this->Size; idx++) {
    float derivNeuronVal = learningRate * this->Errors[idx] * SigmoidDeriv(this->NeuronValues[idx]);
    int weightIndex = idx * this->PreviousLayer->Size;

    for (int j = 0; j < this->PreviousLayer->Size; j++) {
      this->Weights[weightIndex + j] += derivNeuronVal * this->PreviousLayer->NeuronValues[j];
    }

    this->Biases[idx] += learningRate * this->Errors[idx] * SigmoidDeriv(this->NeuronValues[idx]);
  }
}
