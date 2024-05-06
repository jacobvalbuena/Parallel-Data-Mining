package KMeans;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Arrays;

public class SequentialKmeans {
    private int k; //Number of clusters
    private int maxIterations; //Maximum number of iterations
    private double tolerance; //Convergence criterion
    private double[][] centroids;

    public SequentialKmeans(int k, int maxIterations, double tolerance) {
        this.k = k;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
    }

    public void fit(double[][] data) throws Exception {
        centroids = initializeCentroids(data); // Randomly initialize centroids
        boolean convergence = false;
        int iteration = 0;

        //Clustering loop until convergence or maximum iterations
        while (!convergence && iteration < maxIterations) {
            double[][] newCentroids = new double[k][data[0].length];
            List<double[][]> results = new ArrayList<>();

            for (int i = 0; i < k; i++) {
                final int clusterIndex = i;
                //Submits task to assign data points to nearest centroid and calculate sum
                //Parallelizes the assignment step by assigning one thread per cluster
                //Callable<double[][]> task = () -> assignAndSum(data, clusterIndex);
                //results.add(executor.submit(task));
                double[][] task = assignAndSum(data, clusterIndex);
                results.add(task);
            }
            //Collect results and calculate new centroids
            for (int i = 0; i < k; i++) {
                double[][] sumAndCount = results.get(i);
                if (sumAndCount[1][0] > 0) { // Avoid division by zero
                    for (int j = 0; j < data[0].length; j++) {
                        newCentroids[i][j] = sumAndCount[0][j] / sumAndCount[1][0];
                    }
                } else {
                    newCentroids[i] = centroids[i]; // If no points are assigned, retain old centroid
                }
            }

            convergence = checkConvergence(newCentroids);
            centroids = newCentroids;
            iteration++;
        }
    }
    //Randomly select k data points as initial centroids
    private double[][] initializeCentroids(double[][] data) {
        Random random = new Random();
        double[][] initialCentroids = new double[k][data[0].length];
        for (int i = 0; i < k; i++) {
            initialCentroids[i] = data[random.nextInt(data.length)];
        }
        return initialCentroids;
    }
    //Assigns data points to centroid and calculates sum
    private double[][] assignAndSum(double[][] data, int clusterIndex) {
        double[] sum = new double[data[0].length];
        double count = 0;
        for (double[] point : data) {
            if (getNearestCentroid(point) == clusterIndex) {
                for (int j = 0; j < point.length; j++) {
                    sum[j] += point[j];
                }
                count++;
            }
        }
        return new double[][]{sum, new double[]{count}};
    }
    //Returns index of nearest centroid
    private int getNearestCentroid(double[] point) {
        int nearest = -1;
        double minDistance = Double.MAX_VALUE;
        for (int i = 0; i < k; i++) {
            double dist = euclideanDistance(point, centroids[i]);
            if (dist < minDistance) {
                nearest = i;
                minDistance = dist;
            }
        }
        return nearest;
    }
    //Calculates Euclidean distance between two points
    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }
    //Checks if centroids have moved less than tolerance
    private boolean checkConvergence(double[][] newCentroids) {
        for (int i = 0; i < k; i++) {
            if (euclideanDistance(centroids[i], newCentroids[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    //Time Complexity: Sequentially O(n*k*d) where n is the number of data points, k is the number of clusters, and d is the number of dimensions per data point
    //In parallel, the time complexity would be O(n*k*d/p) where p is the number of threads
    //If n = p, the time complexity would be O(k*d)
    public static void main(String[] args) throws Exception {
        double[][] data = {{1, 2}, {1, 5}, {5, 8}, {8, 8}, {1, 0}, {9, 11}, {8, 2}, {10, 2}, {9, 3}};
        SequentialKmeans kMeans = new SequentialKmeans(3, 1000, 0.01);
        kMeans.fit(data);
        System.out.println("Centroids:");
        for (double[] centroid : kMeans.centroids) {
            System.out.println(Arrays.toString(centroid));
        }
    }
}