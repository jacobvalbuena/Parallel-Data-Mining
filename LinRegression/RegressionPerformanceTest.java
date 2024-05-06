package LinRegression;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class RegressionPerformanceTest {

    private static double[][] generateRandomData(int numPoints, int numFeatures) {
        double[][] data = new double[numPoints][numFeatures];
        Random random = new Random();
        for (int i = 0; i < numPoints; i++) {
            for (int j = 0; j < numFeatures; j++) {
                data[i][j] = random.nextDouble() * 100; // Random data scaled up to 100
            }
        }
        return data;
    }

    private static double[] generateRandomLabels(int numPoints) {
        double[] labels = new double[numPoints];
        Random random = new Random();
        for (int i = 0; i < numPoints; i++) {
            labels[i] = random.nextDouble() * 100; // Random labels scaled up to 100
        }
        return labels;
    }

    public static void main(String[] args) {
        int maxDataPoints = 500000; 
        int stepSize = 100000; 
        int numFeatures = 2; 

        
        int d = 10000;
        double[][] Xw = generateRandomData(d, numFeatures);
        double[] yw = generateRandomLabels(d);

        try (FileWriter writer = new FileWriter("LinRegression/performance.csv")) {
            // Write CSV header
            writer.append("DataPoints,SequentialTime,ParallelTime,ParallelNoSplitTime,SequentialSplittingTime\n");

            for (int n = stepSize; n <= maxDataPoints; n += stepSize) {
                // Generate random data
                double[][] X = generateRandomData(n, numFeatures);
                double[] y = generateRandomLabels(n);
                //Run random regression to warm-up
                LinearRegressionSplitting w = new LinearRegressionSplitting(Xw, yw, 4, false);

                // Measure performance of sequential method
                long startTime = System.currentTimeMillis();
                SequentialLinearRegression lr = new SequentialLinearRegression();
                lr.fit(X, y);
                long sequentialTime = System.currentTimeMillis() - startTime;

                // Measure performance of parallel method (splitting then averaging)
                startTime = System.currentTimeMillis();
                LinearRegressionSplitting l = new LinearRegressionSplitting(X, y, 4, false);
                long parallelTime = System.currentTimeMillis() - startTime;

                // Measure performance of parallel method (no splitting)
                startTime = System.currentTimeMillis();
                double[] lrp = LinearRegression.solve(X, y, false);
                long parallelNoSplittingTime = System.currentTimeMillis() - startTime;

                // Measure performance of sequential parallel (sequential + splitting)
                startTime = System.currentTimeMillis();
                LinearRegressionSplitting sp = new LinearRegressionSplitting(X, y, 4, true);
                long sequentialSplittingTime = System.currentTimeMillis() - startTime;

                // Write results to CSV
                writer.append(String.format("%d,%d,%d,%d,%d\n", n, sequentialTime, parallelTime, parallelNoSplittingTime,sequentialSplittingTime));
                writer.flush();
                System.out.println("************************** " + Integer.toString(n) + " Data Points");
                System.out.println(Arrays.toString(lr.getCoefficients()));
                System.out.println(Arrays.toString(l.getCoeffs()));
                System.out.println(Arrays.toString(lrp));
                System.out.println(Arrays.toString(sp.getCoeffs()));
                System.out.println("*************************");
                System.out.println();
            }
            writer.flush();

            
        } catch (IOException e) {
            System.out.println("Error while writing to CSV file: " + e.getMessage());
        }
    }

}
