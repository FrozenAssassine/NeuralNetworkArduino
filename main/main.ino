#include "Layers.h"
#include "NeuralNetwork.h"

void setup() {
  Serial.begin(115200);

  NeuralNetwork *nn = new NeuralNetwork(4);
  nn->StackLayer(new InputLayer(2));
  nn->StackLayer(new DenseLayer(100)); //large numbers just for testing
  nn->StackLayer(new DenseLayer(50)); //large numbers just for testing
  nn->StackLayer(new OutputLayer(1));
  nn->Build();

  float inputs[4][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
  float desired[4][1] = { { 0 }, { 1 }, { 1 }, { 0 } };

  Serial.println("BAGUETTE2");

  nn->TrainCluster((float *)inputs, (float *)desired, 4, 2, 10000, 0.05f, 3);

  //predict stuff:
  for (int i = 0; i < 4; i++) {
    float *pred = nn->Predict(inputs[i], 2);
    Serial.print("PREDICTION ");
    Serial.print(inputs[i][0]);
    Serial.print(" ");
    Serial.print(inputs[i][1]);
    Serial.print(" = ");
    Serial.println(pred[0]);
  }
}

void loop() {
  delay(1000);
}