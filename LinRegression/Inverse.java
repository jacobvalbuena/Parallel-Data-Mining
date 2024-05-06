package LinRegression;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class Inverse {
    private static final ForkJoinPool forkJoinPool = new ForkJoinPool();

    private static class GaussianEliminationTask extends RecursiveAction {
        private double[][] A;
        private double[][] I;
        private int startRow;
        private int endRow;
        private int pivot;

        public GaussianEliminationTask(double[][] A, double[][] I, int startRow, int endRow, int pivot) {
            this.A = A;
            this.I = I;
            this.startRow = startRow;
            this.endRow = endRow;
            this.pivot = pivot;
        }

        @Override
        protected void compute() {
            // Defines the task's execution logic
            // If a task is small enough, it is executed directly
            // Otherwise, it is split into smaller tasks
            if (endRow - startRow <= 10) { // Base case threshold
                for (int i = startRow; i < endRow; i++) {
                    if (i != pivot) {
                        double factor = A[i][pivot] / A[pivot][pivot];
                        for (int j = 0; j < A.length; j++) {
                            A[i][j] -= factor * A[pivot][j];
                            I[i][j] -= factor * I[pivot][j];
                        }
                    }
                }
            } else {
                int mid = (startRow + endRow) / 2;
                invokeAll(new GaussianEliminationTask(A, I, startRow, mid, pivot),
                          new GaussianEliminationTask(A, I, mid, endRow, pivot));
            }
        }
    }

    public static double[][] invert(double[][] A) {
        int n = A.length;
        double[][] I = new double[n][n];
        //Make copy of A to avoid modifying the original matrix
        double [][] B = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                B[i][j] = A[i][j];
            }
        }
        for (int i = 0; i < n; i++) {
            I[i][i] = 1.0;
        }

        // Gaussian elimination
        for (int i = 0; i < n; i++) {
            // Remember that pivot means the first nonzero element in a row
            // If a pivot element is zero, swap the current row with a row below that has a non-zero pivot
            if (B[i][i] == 0) { // Swap rows
                boolean found = false;
                for (int j = i + 1; j < n; j++) {
                    if (B[j][i] != 0) {
                        double[] temp = A[i];
                        B[i] = B[j];
                        B[j] = temp;

                        temp = I[i];
                        I[i] = I[j];
                        I[j] = temp;

                        found = true;
                        break;
                    }
                }
                if (!found) throw new RuntimeException("Matrix is singular and cannot be inverted.");
            }
            // Normalize the pivot row
            // Scale to make the diagonal 1
            double scale = B[i][i];
            for (int j = 0; j < n; j++) {
                B[i][j] /= scale;
                I[i][j] /= scale;
            }

            // Parallel row reduction
            // Perform row reduction in parallel for each pivot
            forkJoinPool.invoke(new GaussianEliminationTask(B, I, 0, n, i));
        }

        // Back substitution is where we are basically solving for the linear system
        // Want to transform B into the identity matrix but also apply the same operations to I
        // I will be the inverse matrix
        for (int col = n - 1; col > 0; col--) {
            for (int row = col - 1; row >= 0; row--) {
                double factor = B[row][col];
                for (int j = 0; j < n; j++) {
                    B[row][j] -= factor * B[col][j];
                    I[row][j] -= factor * I[col][j];
                }
            }
        }
        //Return the inverse matrix
        return I;
        // printMatrix(I); // Method to print the inverse matrix, debugging

        // Time Complexity: In ideal conditions, the time complexity will be less than O(n^3) because we are row reducing in parallel.
        // Realistically, it has O(n^3/p) time complexity where p is the number of processors
        // If you were to have n processors, the time complexity would be O(n^2)
    
    }

    private static void printMatrix(double[][] matrix) {
        for (double[] row : matrix) {
            for (double val : row) {
                System.out.printf("%8.3f", val);
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        // double[][] matrix = {{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
        double[][] matrix = {{1, 0}, {0, 1}};
        // double[][] matrix = {{1, 3, 2, 2}, {2, 3, 1, 1}, {3, 3, 3, 1}, {1, 4, 4, 2}};
        try {
            double[][] I = invert(matrix);
            printMatrix(I); // debugging
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
        

    }
}
