import java.util.Random;

/**
 * Created by thescypion on 3/12/17.
 */
public class Neuron {
    private float value;
    private float weights[];
    private int inputSize;
    private float errorSignal;
    private Neuron input[];


    public Neuron(int inputSize) {
        Random g = new Random();
        this.value = 0;
        this.weights = new float[inputSize];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = g.nextFloat();
        }
        this.inputSize = inputSize;
        this.errorSignal = 0.0f;
    }

    public void calculate(float input[]) {
        float net = 0.0f;
        for (int i = 0; i < inputSize; i++) {
            net += input[i] * weights[i];
        }
        value = activateFunction(net);
    }
    
    public void calculate(Neuron input[]) {
        float net = 0.0f;
        this.input = input;
        for (int i = 0; i < inputSize; i++) {
            net += input[i].getValue() * weights[i];
        }
        value = activateFunction(net);
    }

    public void calculateOutpuError(float t) {
        errorSignal = (t-value) * value * (1-value);
        for (int i = 0; i < inputSize; i++) {
            weights[i] = weights[i] + NeuralNetwork.n * errorSignal * input[i].getValue();
        }
    }

    public void calculateHiddenError(float signal, float weight, float input[]) {
        errorSignal = signal * weight * value;
        for (int i = 0; i < inputSize; i++) {
            weights[i] = weights[i] + NeuralNetwork.n * errorSignal * input[i];
        }
    }

    private float activateFunction(float net) {
        return (float)(1/(1+Math.exp(-net)));
    }

    public float getValue() {
        return value;
    }

    public float[] getWeights() {
        return weights;
    }

    public float getErrorSignal() {
        return errorSignal;
    }
}
