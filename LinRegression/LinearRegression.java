package LinRegression;

public class LinearRegression {
    //Testing the Linear Regression
    public static void main(String[] args) {
        double[][] X = {{1, 1}, {1, 2}, {2, 2}, {2, 3}}; // Example feature matrix
        double[] y = {6, 8, 9, 11}; // Example outcome vector

        solve(X, y, true);
    }
    public static double[] solve(double[][] X, double[] y, boolean print) {
        try{
            double[][] Xt = Transpose.transpose(X); // Step 1: Transpose of X
            // System.out.println("After Transpose");
            // System.out.println(Arrays.deepToString(Xt));

            double[][] XtX = new MatMult(new double[][][] {Xt, X}).solve(); // Step 2: Multiply Xt by X 
            // System.out.println("After XtX Multiplication");
            // System.out.println(Arrays.deepToString(Xt));
            // System.out.println(Arrays.deepToString(XtX));

            double[][] XtX_inv = Inverse.invert(XtX); // Step 3: Invert XtX 
            // System.out.println("After Inversion");
            // System.out.println(Arrays.deepToString(Xt));
            // System.out.println(Arrays.deepToString(XtX));
            // System.out.println(Arrays.deepToString(XtX_inv));

            double[][] XtX_inv_Xt = new MatMult(new double[][][] {XtX_inv, Xt}).solve(); // Step 4: Multiply the inverse by Xt (CORRECT UP TO HERE)
            // System.out.println("After XtX_inv, Xt Multiplication");
            // System.out.println(Arrays.deepToString(Xt));
            // System.out.println(Arrays.deepToString(XtX));
            // System.out.println(Arrays.deepToString(XtX_inv));
            // System.out.println(Arrays.deepToString(XtX_inv_Xt));

            double[] beta = MatMult.multiplyMatrixVector(XtX_inv_Xt, y); // Step 5: Multiply by y vector to get coefficients
            
            if (print) {
                System.out.println("Regression coefficients:");
                for (double b : beta) {
                    System.out.println(b);
                }
            }

            return beta;

        } catch (Exception e) {
            System.out.println(e);
            return null;        
        }
    }
}
