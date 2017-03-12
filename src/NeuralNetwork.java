import java.util.ArrayList;
import java.util.Random;

/**
 * Created by thescypion on 3/11/17.
 */
public class NeuralNetwork {
    private int layersSize[];
    private Neuron hiddenLayer[];
    private Neuron outputLayer[];
    public static float n;
    private float maxError;
    private int maxSteps;
    private int step;

    static {
        n=0.05f;
    }

    public NeuralNetwork(int layersSize[]) {
        this.layersSize = layersSize;

        hiddenLayer = new Neuron[layersSize[1]];
        for (int i = 0; i < layersSize[1]; i++) {
            hiddenLayer[i] = new Neuron(layersSize[0]);
        }

        outputLayer = new Neuron[layersSize[2]];
        for (int i = 0; i < layersSize[2]; i++) {
            outputLayer[i] = new Neuron(layersSize[1]);
        }

        this.maxError = 0.125f;
        this.maxSteps = 1000;
        this.step = 0;
    }

    public void analyze(float input[]) {
        for (Neuron n :
                hiddenLayer) {
            n.calculate(input);
        }

        for (Neuron n :
                outputLayer) {
            n.calculate(hiddenLayer);
        }
    }

    public void learn(float P[][], float T[][]) {
        for (int c = 0; c < maxSteps; c++) {
            float error = 0.0f;
            for (int l = 0; l < P.length; l++) {
                analyze(P[l]);
                //error output layer
                for (int i = 0; i < layersSize[2]; i++) {
                    outputLayer[i].calculateOutpuError(T[l][i]);
                }
                //error hidden layer
                for (int i = 0; i < layersSize[1]; i++) {
                    for (Neuron n : outputLayer) {
                        float w = n.getWeights()[i];
                        hiddenLayer[i].calculateHiddenError(n.getErrorSignal(), w, P[l]);
                    }
                }
                float factor = 0.0f;
                for (int i = 0; i < layersSize[2]; i++) {
                    factor+=(outputLayer[i].getValue()-T[l][i]) * (outputLayer[i].getValue()-T[l][i]);
                }
                error = error + 0.5f * factor;

            }
            if(error < maxError) break;
        }

    }

    public void display() {
        for (Neuron n :
                outputLayer) {
            System.out.println(n.getValue());
        }
    }

}
