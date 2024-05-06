import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.io.BufferedReader;
import java.io.FileReader;

public class RegressionPerformanceTest {

    public static void main(String[] args) {
        int maxDataPoints = 100000; // Maximum number of data points to test
        int stepSize = 10000; // Step size for increasing data points
        int numFeatures = 2; // Number of features in the dataset

        try (FileWriter writer = new FileWriter("performance.csv")) {
            // Write CSV header
            writer.append("DataPoints,SequentialTime,ParallelTime\n");

            for (int n = stepSize; n <= maxDataPoints; n += stepSize) {
                // Generate random data
                double[][] X = generateRandomData(n, numFeatures);
                double[] y = generateRandomLabels(n);

                // Measure performance of sequential method
                long startTime = System.currentTimeMillis();
                SequentialLinearRegression lr = new SequentialLinearRegression();
                lr.fit(X, y);
                long sequentialTime = System.currentTimeMillis() - startTime;

                // Measure performance of parallel method
                startTime = System.currentTimeMillis();
                // double[] beta = LinearRegression.solve(X, y, false);
                BufferedReader br = new BufferedReader(new FileReader("./Student_Performance.csv"));
                String curLine;
                double[][] x = new double[10000][5];
                double[] y1 = new double[10000];
                int i = 0;
                while ((curLine = br.readLine()) != null && i < x.length) {
                    String[] values = curLine.split(",");
                    double[] valuesD = new double[values.length - 1];
                    for (int j = 0; j < x[0].length; j++) {
                        if (values[j].equals("Yes")) {
                            valuesD[j] = 1;
                        } else if (values[j].equals("No")) {
                            valuesD[j] = 0;
                        } else {
                            valuesD[j] = Double.parseDouble(values[j]);
                        }
                    }
                    y[i] = Double.parseDouble(values[values.length - 1]);
                    x[i] = valuesD;
                    i++;
                }
                try{
                    LinearRegressionSharding l = new LinearRegressionSharding(x, y1, 4);
                } catch (Exception e) {
                    System.out.println(e);
                }
                long parallelTime = System.currentTimeMillis() - startTime;

                // Write results to CSV
                writer.append(String.format("%d,%d,%d\n", n, sequentialTime, parallelTime));
                writer.flush();
            }
            writer.flush();
        } catch (IOException e) {
            System.out.println("Error while writing to CSV file: " + e.getMessage());
        }
    }

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
}
