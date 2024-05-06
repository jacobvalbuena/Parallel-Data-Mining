package LinRegression;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class LinearRegressionSplitting {
    // Input will be each line of the x file will be a 
    private double[][] x;
    private double[] y;
    private double[][] coefs;
    private double[] finalCoefs;
    
    // Constructor will add x and y to appropriate variables from a csv file, calc regression coefficients for each split, and 
    public LinearRegressionSplitting(double[][] x, double[] y, int maxShards, boolean sequential) {
        try {
            this.x = x;
            this.y = y;
            coefs = new double[maxShards][x[0].length];
            ExecutorService executorService = Executors.newFixedThreadPool(Math.min(Runtime.getRuntime().availableProcessors(), maxShards));

            for (int i = 0; i < maxShards; i++) {
                int j = i;
               executorService.execute(() -> {
                    coefs[j] = getCoefficientsSplit(maxShards, j, sequential);
                    }
                );
            } 
            executorService.shutdown();
            executorService.awaitTermination(10, TimeUnit.SECONDS);

        } catch (Exception e) {}

        finalCoefs = new double[coefs[0].length];
        //Average to get final coefficients
        // Fill with 0
        Arrays.fill(finalCoefs, 0);
        // For each set of coefficients (splits)
        for (int i = 0; i < maxShards; i++) {
            // for each feature coefficient in a split
            for (int j = 0; j < coefs[i].length; j++) {
                finalCoefs[j] += coefs[i][j];
            }
        }

        //Divide
        for (int i = 0; i < finalCoefs.length; i++) {
            finalCoefs[i] /= maxShards;
        }
    }

    // Will calculate error for split num
    public double calcError(String fileNameString) {
        double error = 0;
        // FOr each x point
        for (int i = 0; i < x.length; i++) {
            double value = 0;
            //for each feature
            for (int j = 0; j < x[i].length; j++) {
                value += x[i][j] * finalCoefs[j];
            }
            error += Math.abs(value - y[i]);
        }
        return error;
    }

    private double[] getCoefficientsSplit(int numSplits, int splitNum, boolean sequential) {
        try {
            int start = splitNum * (x.length/numSplits);
            int end = Math.min((splitNum + 1) * (x.length/numSplits), x.length);
            double[][] splitX = Arrays.copyOfRange(this.x, start, end);
            double [] splitY = Arrays.copyOfRange(this.y, start, end);
            if (!sequential) {
                double[] betas = LinearRegression.solve(splitX, splitY, false);
                return betas;
            } else {
                SequentialLinearRegression s = new SequentialLinearRegression();
                s.fit(splitX, splitY);
                return s.getCoefficients();
            }

            
        } catch (Exception e) {
            return null;
        }
    }

    public double[] getCoeffs() {
        return this.finalCoefs;
    }
}
