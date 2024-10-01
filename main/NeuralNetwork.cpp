#include "NeuralNetwork.h"
#include "Arduino.h"
#include <Wire.h>

#define SLAVE_1_ADDRESS 0x08
#define SLAVE_2_ADDRESS 0x09
#define SLAVE_3_ADDRESS 0x0A

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

float* NeuralNetwork::Predict(float* inputs, int inputLength) {
  //give the input neurons the input values:
  for (int j = 0; j < inputLength; j++) {
    this->allLayer[0]->NeuronValues[j] = inputs[j];
  }

  //Feed forward input values:
  for (int j = 1; j < this->totalLayers; j++) {
    this->allLayer[j]->FeedForward();
  }

  return this->allLayer[this->totalLayers - 1]->NeuronValues;
}

void NeuralNetwork::TrainCluster(float* inputs, float* desired, int totalItems, int inputItemCount, int epochs, float learningRate, int numSlaves) {
  Serial.println("Begin distributed training with ESP32 Cluster");

  for (int epoch = 0; epoch < epochs; epoch++) {
    for (int i = 0; i < totalItems; i++) {

      // Feed forward the input values:
      for (int j = 0; j < inputItemCount; j++) {
        this->allLayer[0]->NeuronValues[j] = inputs[i * inputItemCount + j];
      }

      for (int j = 1; j < this->totalLayers; j++) {
        this->allLayer[j]->FeedForward();
      }

      // Backpropagate and distribute weight updates across multiple ESPs
      for (int j = this->totalLayers - 1; j >= 0; j--) {
        if (j == this->totalLayers - 1) {
          // Handle the output layer locally
          this->allLayer[j]->Train(&desired[i], learningRate);
        } else {
          // Distribute calculation of the hidden layers
          int layerSize = this->allLayer[j]->Size;
          int chunkSize = layerSize / numSlaves;

          // Send chunks to each slave
          for (int slave = 0; slave < numSlaves; slave++) {
            int startIdx = slave * chunkSize;
            int endIdx = (slave == numSlaves - 1) ? layerSize : startIdx + chunkSize;

            // Send weights, biases, and errors to the slave for processing
            sendLayerDataToSlave(SLAVE_1_ADDRESS + slave, j, startIdx, endIdx, learningRate);
          }

          // Receive the updated weights and biases from each slave
          for (int slave = 0; slave < numSlaves; slave++) {
            receiveLayerDataFromSlave(SLAVE_1_ADDRESS + slave, j);
          }
        }
      }
    }

      Serial.print("EPOCH ");
      Serial.println(epoch);
  }

  Serial.println("Distributed Training Done!");
}

void NeuralNetwork::sendLayerDataToSlave(int slaveAddress, int layerIndex, int startIdx, int endIdx, float learningRate) {
  Wire.beginTransmission(slaveAddress);
  Wire.write(layerIndex); // Send layer index
  Wire.write((byte*)&learningRate, sizeof(learningRate)); // Send learning rate

  for (int i = startIdx; i < endIdx; i++) {
    Wire.write((byte*)&this->allLayer[layerIndex]->Weights[i], sizeof(float));
    Wire.write((byte*)&this->allLayer[layerIndex]->Biases[i], sizeof(float));
    Wire.write((byte*)&this->allLayer[layerIndex]->Errors[i], sizeof(float));
  }

  Wire.endTransmission();
  delay(10); // Small delay for stability
}

void NeuralNetwork::receiveLayerDataFromSlave(int slaveAddress, int layerIndex) {
  int layerSize = this->allLayer[layerIndex]->Size;
  Wire.requestFrom(slaveAddress, sizeof(float) * layerSize * 2); // Request weights and biases

  for (int i = 0; i < layerSize; i++) {
    Wire.readBytes((byte*)&this->allLayer[layerIndex]->Weights[i], sizeof(float));
    Wire.readBytes((byte*)&this->allLayer[layerIndex]->Biases[i], sizeof(float));
  }
}

void NeuralNetwork::Train(float* inputs, float* desired, int totalItems, int inputItemCount, int epochs, float learningRate) {
  Serial.println("Begin training");

  for (int epoch = 0; epoch < epochs; epoch++) {
    for (int i = 0; i < totalItems; i++) {

      //feed forward the input values:
      for (int j = 0; j < inputItemCount; j++) {
        this->allLayer[0]->NeuronValues[j] = inputs[i * inputItemCount + j];
      }

      for (int j = 1; j < this->totalLayers; j++) {
        this->allLayer[j]->FeedForward();
      }

      //back propagate the model:
      for (int j = this->totalLayers - 1; j >= 0; j--) {
        this->allLayer[j]->Train(&desired[i], learningRate);
      }
    }

    if (epoch % 100 == 0) {
      Serial.print("EPOCH ");
      Serial.println(epoch);
    }
  }

  Serial.println("Training Done!");
}