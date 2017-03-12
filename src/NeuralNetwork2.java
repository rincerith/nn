import java.util.Random;

/**
 * Created by thescypion on 3/12/17.
 */
public class NeuralNetwork2 {
    private int inputSize;
    private float input[];

    private int hiddenSize;
    private float hiddenValues[];
    private float hiddenWeights[][]; //hidden input
    private float hiddenError[];


    private int outputSize;
    private float outputValues[];
    private float outputWeights[][]; //output hidden
    private float outputError[];

    private float N;
    private int maxStep;
    private float maxError;

    public NeuralNetwork2(int inputSize, int hiddenSize, int outputSize, float N, int maxStep, float maxError) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.N = N;
        this.maxStep = maxStep;
        this.maxError = maxError;

        Random random = new Random();

        hiddenValues = new float[hiddenSize];
        hiddenWeights = new float[hiddenSize][inputSize];
        for (int h_i = 0; h_i < hiddenSize; h_i++) {
            for (int i_i = 0; i_i < inputSize; i_i++) {
                hiddenWeights[h_i][i_i] = random.nextFloat() - 0.5f;
            }
        }
        hiddenError = new float[hiddenSize];

        outputValues = new float[outputSize];
        outputWeights = new float[outputSize][hiddenSize];
        for (int o_i = 0; o_i < outputSize; o_i++) {
            for (int h_i = 0; h_i < hiddenSize; h_i++) {
                outputWeights[o_i][h_i] = random.nextFloat() - 0.5f;
            }
        }
        outputError = new float[outputSize];
    }

    public void analyze(float input[]) {
        this.input = input;

        for (int h_i = 0; h_i < hiddenSize; h_i++) {
            float net = 0.0f;
            for (int i_i = 0; i_i < inputSize; i_i++) {
                net += input[i_i] * hiddenWeights[h_i][i_i];
            }
            hiddenValues[h_i] = activateFunction(net);
        }

        for (int o_i = 0; o_i < outputSize; o_i++) {
            float net = 0.0f;
            for (int h_i = 0; h_i < hiddenSize; h_i++) {
                net += hiddenValues[h_i] * outputWeights[o_i][h_i];
            }
            outputValues[o_i] = activateFunction(net);
        }
    }

    public void learn(float P[][], float T[][]) {
        int TEST = 0;

        for (int step = 0; step < maxStep; step++) {
            float error = 0.0f;
            for (int p_i = 0; p_i < P.length; p_i++) {
                analyze(P[p_i]);

                for (int o_i = 0; o_i < outputSize; o_i++) {
                    outputError[o_i] = (T[TEST][o_i] - outputValues[o_i]) * (1 - outputValues[o_i]);
                    for (int h_i = 0; h_i < hiddenSize; h_i++) {
                        outputWeights[o_i][h_i] += N * outputError[o_i] * hiddenValues[h_i];
                    }
                }

                for (int h_i = 0; h_i < hiddenSize; h_i++) {
                    for (int o_i = 0; o_i < outputSize; o_i++) {
                        for (int i_i = 0; i_i < inputSize; i_i++) {
                            hiddenError[h_i] = outputError[o_i] * hiddenWeights[h_i][i_i] * input[i_i];
                            hiddenWeights[h_i][i_i] += N * hiddenError[h_i] * input[i_i];
                        }
                    }
                }

                float factor = 0.0f;
                for (int o_i = 0; o_i < outputSize; o_i++) {
                    float x = outputValues[o_i] - T[p_i][o_i];
                    factor+= x*x*0.5f;
                }
                error+=factor;
            }
            if(error < maxError) break;
        }


    }

    private float activateFunction(float net) {
        return (float)(1/(1+Math.exp(-net)));
    }

    public void display() {
        for (float v : outputValues) {
            System.out.println(v);
        }
    }

}
