<!--<p align="center">
    <img src="path_to_your_logo" height="150px" width="auto" alt="Neural Network Logo">
</p>
-->

<h1 align="center">Neural Network for ESP32 and Arduino</h1>
<div align="center">
    <img src="https://img.shields.io/github/stars/FrozenAssassine/NeuralNetworkArduino?style=flat"/>
    <img src="https://img.shields.io/github/issues-pr/FrozenAssassine/NeuralNetworkArduino?style=flat"/>
    <img src="https://img.shields.io/github/repo-size/FrozenAssassine/NeuralNetworkArduino?style=flat"/>
</div>

## 🤔 What is this project?

This project is a lightweight neural network implementation designed to run on microcontrollers like the **ESP32** and **Arduino**. It demonstrates how even resource-constrained devices can train and perform simple tasks like **XOR** prediction.

While it takes just some **seconds** to train on the ESP32, the Arduino requires significantly more time due to limited processing power.

## 🛠️ Features
- **On-device training**: Train your neural network directly on ESP32 or Arduino.
- **XOR**: Predict simple numbers like in xor.
- **Fast Training**: The ESP32 can train in just a few seconds, while the Arduino requires longer due to its slow processor.

## 🔮 Future features
- Train on PC and load weights to chip
- Save and load weights
- More layer types
- Simple performance optimizations
- Use multiple threads on esp32

## 🚀 Performance
- ESP32: Fast training (~seconds).
- Arduino: Slower training (~minutes or more).

## 🏗️ How to Use

1. Clone this repository and open the project in your preferred environment for ESP32 or Arduino.
2. Upload the code to your ESP32 or Arduino using the Arduino IDE or another compatible tool.
3. Monitor the predictions via Serial Monitor at 115200 baud rate.

Here is an example code:

```cpp
#include "Layers.h"
#include "NeuralNetwork.h"

void setup() {
  Serial.begin(115200);

  NeuralNetwork *nn = new NeuralNetwork(3);
  nn->StackLayer(new InputLayer(2));
  nn->StackLayer(new DenseLayer(4));
  nn->StackLayer(new OutputLayer(1));
  nn->Build();

  float inputs[4][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
  float desired[4][1] = { { 0 }, { 1 }, { 1 }, { 0 } };

  nn->Train((float*)inputs, (float*)desired, 4, 2, 10000, 0.05f);

  // Predict XOR results:
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
```

# 📷 Images:
![image](https://github.com/user-attachments/assets/a84f9521-d4cc-482d-92d5-03c0d8f58a3d)

