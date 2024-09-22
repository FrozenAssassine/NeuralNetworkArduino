#ifndef LAYERS_H
#define LAYERS_H

enum class ActivationType {
    SIGMOID
};

float Sigmoid(float x);
float SigmoidDeriv(float x);

void FillRandom(float *array, int size);

class BaseLayer {
public:
    float* Biases;
    float* NeuronValues;
    float* Errors;
    float* Weights;
    int Size;
    BaseLayer* PreviousLayer;
    BaseLayer* NextLayer;
    ActivationType ActivationFunction;

    BaseLayer();
    virtual ~BaseLayer();

    virtual void FeedForward() = 0;
    virtual void Train(const float* desiredValues, float learningRate) = 0;
    virtual void InitLayer(int size, BaseLayer* previous, BaseLayer* next) = 0;
};


class DenseLayer : public BaseLayer {
public:
    DenseLayer(int size);
    ~DenseLayer();
    void InitLayer(int size, BaseLayer* previous, BaseLayer* next) override;
    void FeedForward() override;
    void Train(const float* desiredValues, float learningRate) override;
};


class InputLayer : public BaseLayer {
public:
    InputLayer(int size);
    ~InputLayer();
    void InitLayer(int size, BaseLayer* previous, BaseLayer* next) override;
    void FeedForward() override;
    void Train(const float* desiredValues, float learningRate) override;
};


class OutputLayer : public BaseLayer {
public:
    OutputLayer(int size);
    ~OutputLayer();
    void InitLayer(int size, BaseLayer* previous, BaseLayer* next) override;
    void FeedForward() override;
    void Train(const float* desiredValues, float learningRate) override;
};

#endif