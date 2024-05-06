import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;


public class SequentialLinearRegression {
    private RealMatrix coefficients;

    public void fit(double[][] X, double[] y) {
        // Convert arrays to RealMatrix
        RealMatrix matrixX = new Array2DRowRealMatrix(X, true);
        RealMatrix matrixY = new Array2DRowRealMatrix(y);

        // Adding an intercept term (column of ones)
        matrixX = extendMatrixWithIntercept(matrixX);

        // Regularization factor (lambda)
        double lambda = 0.01;
        RealMatrix XTX = matrixX.transpose().multiply(matrixX);

        // Add lambda * I to X^T * X to ensure it's invertible
        RealMatrix lambdaI = MatrixUtils.createRealIdentityMatrix(XTX.getRowDimension()).scalarMultiply(lambda);
        RealMatrix regularizedXTX = XTX.add(lambdaI);

        RealMatrix XTY = matrixX.transpose().multiply(matrixY);
        DecompositionSolver solver = new LUDecomposition(regularizedXTX).getSolver();

        coefficients = solver.solve(XTY);
    }

    private RealMatrix extendMatrixWithIntercept(RealMatrix matrix) {
        int numRows = matrix.getRowDimension();
        int numCols = matrix.getColumnDimension() + 1; // Adding one for the intercept term
        RealMatrix extendedMatrix = new Array2DRowRealMatrix(numRows, numCols);
        extendedMatrix.setSubMatrix(matrix.getData(), 0, 1);
        extendedMatrix.setColumnVector(0, new Array2DRowRealMatrix(new double[numRows]).getColumnVector(0)); // Add 1s for intercept
        return extendedMatrix;
    }
    //
    public double[] predict(double[][] X) {
        RealMatrix matrixX = new Array2DRowRealMatrix(X, true);
        matrixX = extendMatrixWithIntercept(matrixX);
        RealMatrix predictions = matrixX.multiply(coefficients);
        return predictions.getColumn(0);
    }

    public double[] getCoefficients() {
        return coefficients.getColumn(0);
    }

    public static void main(String[] args) {
        double[][] X = {{1, 2}, {2, 3}, {4, 5}, {5, 6}};
        double[] y = {1, 2, 3, 4};

        SequentialLinearRegression lr = new SequentialLinearRegression();
        lr.fit(X, y);
        System.out.println("Coefficients:");
        for (double coef : lr.getCoefficients()) {
            System.out.println(coef);
        }

        double[][] testData = {{1, 2}, {3, 4}};
        double[] predictions = lr.predict(testData);
        System.out.println("Predictions:");
        for (double pred : predictions) {
            System.out.println(pred);
        }
    }
}
