package LinRegression;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;



public class MatMult {
    private double[][][] matrices;

    // Initialize a three-dimensional array that holds two matrices
    public MatMult(double matrices[][][]) {
        this.matrices = matrices;
        
    }

    private double iterateMult(double[][] m1, double[][] m2, int rowNum, int colNum) {
        int k = m1[0].length; // Should match the number of rows in m2
        if (k != m2.length) {
            throw new IllegalArgumentException("Matrix dimensions do not match for multiplication.");
        }
        // Multiply the row of the first matrix by the column of the second matrix
        double sum = 0;
        for (int i = 0; i < k; i++) {
            sum += m1[rowNum][i] * m2[i][colNum];
        }
    
        return sum;
    }
    
    // Time Complexity: Sequentially O(n1*m2*k) where n1 is the number of rows in the first matrix, m2 is the number of columns in the second matrix, and k is the number of columns in the first matrix (or rows in the second matrix).
    // Theoretically, it could be O(k) if you were to have n1*m2 processors
    // Every element of the result matrix is computed independently
    public double[][] multiplyTwo(double[][] mat1, double[][] mat2) throws Exception {
        int n1 = mat1.length;
        int m1 = mat1[0].length;
        int n2 = mat2.length;
        int m2 = mat2[0].length;
        ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        // Check if the matrices can be multiplied
        if (m1 != n2) {
            throw new IllegalArgumentException("Matrix dimensions must match for multiplication.");
        }
    
        double[][] result = new double[n1][m2];
        // Each thread computes one element of the result matrix
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < m2; j++) {
                int row = i;
                int col = j;
                executorService.execute(() -> {
                    result[row][col] = iterateMult(mat1, mat2, row, col);
                });
            }
        }

        executorService.shutdown();
        executorService.awaitTermination(10, TimeUnit.SECONDS);
    
        return result;
    }
    
    public static double[] multiplyMatrixVector(double[][] matrix, double[] vector) throws Exception {
        int numRows = matrix.length;
        int numCols = matrix[0].length;
        // Check if the matrix and vector dimensions match
        if (numCols != vector.length) {
            throw new IllegalArgumentException("Matrix and vector dimensions must match.");
        }
    
        double[] result = new double[numRows];
        ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        // Each thread computes one element of the result vector
        for (int i = 0; i < numRows; i++) {
            final int row = i;
            executorService.execute(() -> {
                double sum = 0;
                for (int j = 0; j < numCols; j++) {
                    sum += matrix[row][j] * vector[j];
                }
                result[row] = sum;
            });
            
        }
    
        executorService.shutdown();
        executorService.awaitTermination(10, TimeUnit.SECONDS);
    
        return result;
    }
    
    public double[][] solve() throws Exception{
        return multiplyTwo(matrices[0], matrices[1]);
    }

    // Testing method
    public static void main(String[] args) throws Exception{
        double[][] m1 = new double[][] {
            {1, 1, 2, 2},
            {1, 2, 2, 3}
        };

        double[][] m2 = new double[][] {
            {1, 1}, 
            {1, 2}, 
            {2, 2}, 
            {2, 3}
        };

        double[][] result = new MatMult(new double[][][] {m1, m2}).solve();
        for (double[] row: result) {
            System.out.println(Arrays.toString(row));
        }
    }
}
