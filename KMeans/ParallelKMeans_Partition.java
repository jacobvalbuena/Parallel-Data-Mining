package KMeans;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.Random;
import java.util.Arrays;

public class ParallelKMeans_Partition {
    private int k; // Number of clusters
    private int maxIterations; // Maximum number of iterations
    private double tolerance; // Convergence criterion
    private ExecutorService executor;
    private double[][] centroids; // To store the centroids
    private int numThreads;  // To store the number of threads

    public ParallelKMeans_Partition(int k, int maxIterations, double tolerance, int numThreads) {
        this.k = k;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.executor = Executors.newFixedThreadPool(numThreads);
        this.numThreads = numThreads;  // Initialize numThreads
    }

    public void fit(double[][] data) throws Exception {
        List<List<double[]>> partitions = partitionData(data, numThreads);
        // Partitions the data into numThreads partitions
        centroids = initializeCentroids(data);

        boolean convergence = false;
        int iteration = 0;
        // Clustering loop until convergence or maximum iterations
        while (!convergence && iteration < maxIterations) {
            List<Future<double[][][]>> results = new ArrayList<>();
            double[][] newCentroids = new double[k][data[0].length];

            for (List<double[]> partition : partitions) {
                final List<double[]> finalPartition = partition;
                // Submits task to assign data points to nearest centroid and calculate sum
                // Parallelizes the assignment step by assigning one thread per partition
                Callable<double[][][]> task = () -> assignAndSum(finalPartition);
                results.add(executor.submit(task));
            }
            
            double[][] globalSums = new double[k][data[0].length];
            double[] globalCounts = new double[k];
            // Retrieve results of partitions
            for (Future<double[][][]> future : results) {
                double[][][] localResults = future.get();
                for (int i = 0; i < k; i++) {
                    globalCounts[i] += localResults[1][0][i];
                    for (int j = 0; j < data[0].length; j++) {
                        globalSums[i][j] += localResults[0][i][j];
                    }
                }
            }
            // Calculate new centroids
            for (int i = 0; i < k; i++) {
                if (globalCounts[i] > 0) {
                    for (int j = 0; j < data[0].length; j++) {
                        newCentroids[i][j] = globalSums[i][j] / globalCounts[i];
                    }
                } else {
                    newCentroids[i] = centroids[i]; // Retain old centroid if no points assigned
                }
            }
            // Check convergence
            convergence = checkConvergence(newCentroids);
            centroids = newCentroids;
            iteration++;
        }
        executor.shutdown();
    }
    //Partitions the data into numPartitions partitions
    private List<List<double[]>> partitionData(double[][] data, int numPartitions) { 
        List<List<double[]>> partitions = new ArrayList<>();
        for (int i = 0; i < numPartitions; i++) {
            partitions.add(new ArrayList<>());
        }
        for (int i = 0; i < data.length; i++) {
            partitions.get(i % numPartitions).add(data[i]);
        }
        return partitions;
    }
    //For each partition, assigns data points to nearest centroid and calculates sum
    private double[][][] assignAndSum(List<double[]> partition) {
        double[][] sum = new double[k][partition.get(0).length];
        double[] count = new double[k];
        for (double[] point : partition) {
            int nearest = getNearestCentroid(point);
            count[nearest]++;
            for (int j = 0; j < point.length; j++) {
                sum[nearest][j] += point[j];
            }
        }
        return new double[][][]{sum, new double[][]{count}};
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
    //Checks if the algorithm has converged
    private boolean checkConvergence(double[][] newCentroids) {
        for (int i = 0; i < k; i++) {
            if (euclideanDistance(centroids[i], newCentroids[i]) > tolerance) {
                return false;
            }
        }
        return true;
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
    //Time Complexity: Sequentially O(n*k*d) where n is the number of data points, k is the number of clusters, and d is the number of dimensions per data point
    //Parallely O(n*k*d/p) where p is the number of partitions
    //If p is tailored to handle the number of centroids, effectively matching how many data points each thread will handle,
    //the time complexity will be O(k*d)
    public static void main(String[] args) throws Exception {
        double[][] data = {{1, 2}, {1, 5}, {5, 8}, {8, 8}, {1, 0}, {9, 11}, {8, 2}, {10, 2}, {9, 3}};
        ParallelKMeans_Partition kMeans = new ParallelKMeans_Partition(3, 1000, 0.01, 4);
        kMeans.fit(data);
        System.out.println("Centroids:");
        for (double[] centroid : kMeans.centroids) {
            System.out.println(Arrays.toString(centroid));
        }
    }
}
