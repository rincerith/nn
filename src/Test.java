import java.util.Random;

/**
 * Created by thescypion on 3/11/17.
 */
public class Test {
    public static void main(String[] args) {
        Random generator = new Random();

        NeuralNetwork2 nn = new NeuralNetwork2(3, 2, 1, 0.05f, 1000, 0.1f);

        float P[][] = new float[][]{
            {-1, -1, 1},
            {-1, 1, 1},
            {1, -1, 1},
            {1, 1, 1}
        };

        float T[][] = new float[][]{{0}, {1}, {1}, {0}};

        nn.learn(P, T);

        for (int i = 0; i < P.length; i++) {
            System.out.println(i);
            nn.analyze(P[i]);
            nn.display();
        }
    }
}
