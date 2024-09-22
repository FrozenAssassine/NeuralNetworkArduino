#include "Layers.h"
#include "NeuralNetwork.h"

void setup() {
  Serial.begin(115200);
  Serial.println("Start");


  InputLayer* inLayer = new InputLayer(2);
  DenseLayer* hiddenLayer = new DenseLayer(4);
  OutputLayer* outLayer = new OutputLayer(4);

  inLayer->InitLayer(2, nullptr, hiddenLayer);
  hiddenLayer->InitLayer(4, inLayer, outLayer);
  outLayer->InitLayer(1, hiddenLayer, nullptr);


  float inputs[4][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
  float desired[4][1] = { { 0 }, { 1 }, { 1 }, { 0 } };

  Serial.println("Begin training");

  // nn->Train((float*)inputs, (float*)desired, 4, 20000, 0.01f);
  float learningRate = 0.01f;
  for (int epoch = 0; epoch < 15900; epoch++) {
    for (int i = 0; i < 4; i++) {
      //TODO: hardcoded index is not good!
      inLayer->NeuronValues[0] = inputs[i][0];
      inLayer->NeuronValues[1] = inputs[i][1];

      hiddenLayer->FeedForward();
      outLayer->FeedForward();

      outLayer->Train(desired[i], learningRate);
      hiddenLayer->Train(desired[i], learningRate);
    }

    if (epoch % 100 == 0) {
      Serial.print("EPOCH ");
      Serial.println(epoch);
    }
  }

  Serial.println("Training Done!");

  for (int i = 0; i < 4; i++) {
    inLayer->NeuronValues[0] = inputs[i][0];
    inLayer->NeuronValues[1] = inputs[i][1];

    hiddenLayer->FeedForward();
    outLayer->FeedForward();

    Serial.print("Predicted: ");
    Serial.println(outLayer->NeuronValues[0]);
  }
}

void loop() {
  delay(1000);
}