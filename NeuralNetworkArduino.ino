#include "Layers.h"
#include "NeuralNetwork.h"

void setup() {
  Serial.begin(115200);
  Serial.println("Start");

  NeuralNetwork *nn = new NeuralNetwork(3);
  nn->StackLayer(new InputLayer(2));
  nn->StackLayer(new DenseLayer(4));
  nn->StackLayer(new OutputLayer(1));
  nn->Build();

  float inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  float desired[4][1] = {{0}, {1}, {1}, {0}};

  Serial.println("Begin training");

  // nn->Train((float*)inputs, (float*)desired, 4, 20000, 0.01f);
  float learningRate = 0.01f;
  for (int epoch = 0; epoch < 15900; epoch++) {
    for (int i = 0; i < 4; i++) {
      //TODO: hardcoded index is not good!
      nn->allLayer[0]->NeuronValues[0] = inputs[i][0];
      nn->allLayer[0]->NeuronValues[1] = inputs[i][1];

      for (int j = 0; j < nn->totalLayers; j++) {
        nn->allLayer[j]->FeedForward();
      }
      for (int j = nn->totalLayers - 1; j >= 0; j--) {
        nn->allLayer[j]->Train(desired[i], learningRate);
      }
    }

    if (epoch % 100 == 0) {
      Serial.print("EPOCH ");
      Serial.println(epoch);
    }
  }


  Serial.println("Training Done!");

  BaseLayer * inputLayer = nn->allLayer[0];
  for(int i = 0; i<4; i++){
    inputLayer->NeuronValues[0] = inputs[i][0];
    inputLayer->NeuronValues[1] =  inputs[i][1];

    Serial.print("Predicted: ");
    Serial.println(nn->allLayer[2]->NeuronValues[0]);
  }
}

void loop() {
  delay(1000);
}