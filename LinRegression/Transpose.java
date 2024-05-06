package LinRegression;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class Transpose {
    public static void main(String[] args) {
        double[][] matrix = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };
        
        double[][] transposedMatrix = transpose(matrix);
        
        for (double[] row : transposedMatrix) {
            for (double val : row) {
                System.out.print(val + " ");
            }
            System.out.println();
        }
    }
    //Time complexity: Sequentially O(n^2) 
    //if you were to have n processors, the time complexity would be O(n)
    public static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposed = new double[cols][rows];
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        for (int i = 0; i < rows; i++) {
            int finalI = i;
            // Loop through each row of the original matrix 
            // Submit a task to the executor to transpose the row
            executor.submit(() -> {
                for (int j = 0; j < cols; j++) {
                    transposed[j][finalI] = matrix[finalI][j];
                }
            });
        }

        executor.shutdown();
        try {
            if (!executor.awaitTermination(1, TimeUnit.MINUTES)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }

        return transposed;
    }
}
