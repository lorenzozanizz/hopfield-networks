#include <vector>

#include "mappings/konohen_mapping_eigen.hpp"
#include "mappings/classifier/majority_mapping_eigen.hpp"
#include "mappings/clustering/u_clustering_eigen.hpp"
#include "mappings/clustering/cluster.hpp"
#include "mappings/clustering/k_means.hpp"

#include "io/plot/plot.hpp"
#include "io/datasets/dataset.hpp"
#include "io/datasets/repository.hpp"

int main() {

	using namespace Eigen;
	using DoubleVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

	// We again use MNIST to evaluate the ability of the konohen mappings to
	// create a semantically significant map between the input cortex, e.g. the
	// distiribution of MNIST digits, and a mapping cortex represented by a 
	// discrete grid of units. 
	constexpr const auto SAMPLE_SIZE = 4000;

	VectorDataset<DoubleVector, unsigned int> mnist(SAMPLE_SIZE);
	DatasetRepo::load_mnist_eigen("vector_mnist_full.data", SAMPLE_SIZE, mnist);

	constexpr const auto MNIST_SIZE = 28;

	unsigned int map_width = 10;
	unsigned int map_height = 10;
	// We create the mapping assigning it a width and height, and report the size
	// of the input cortex. The mapping will associate items from the input distribution
	// to winning units in the width x height lattice map
	KonohenMapEigen<double> km(map_width, map_height, MNIST_SIZE * MNIST_SIZE);

	// Evolving function parameters (how the sigma changes through the iterations)
	// Other options: linear, piecewise and inverse_time 
	std::string evolving_func = "exponential"; 
	double sigma0 = 3.0;
	double tau = 10.0;
	// used in the piecewise evolving function
	double sigma1 = 1.0; 
	// used in the inverse_time method
	double beta = 0.0;

	// We perform 100 iterations of the method, in which we iterate over batches in the
	// dataset and learn the input distribution. 
	unsigned int iterations = 100; 

	NeighbouringFunctionEigen nf(sigma0, tau, map_width, map_height, evolving_func);
	nf.set_sigma_1(sigma1);
	nf.set_beta(beta);
	nf.set_t_max(iterations);

	// Initializing and training Konohen map
	nf.set_support_size(2);
	km.initialize();

	double learning_rate = 0.10; // weight of the update in each iteration in the Kohonen map
	km.train(mnist, iterations, nf, learning_rate);

	// Creating the label map for MNIST
	std::map<int, std::string> labels_map;
	// Initialize the units
	labels_map[0] = "unknown";	labels_map[1] = "zero";		labels_map[2] = "one";
	labels_map[3] = "two";		labels_map[4] = "three";	labels_map[5] = "four";
	labels_map[6] = "five";		labels_map[7] = "six";		labels_map[8] = "seven";
	labels_map[9] = "eight";	labels_map[10] = "nine";

	// Majority map parameters
	double threshold = 0.6; // the percentage of hits required to label a neuron in the classifier
	// Initializing the majority map and performing the classification
	MajorityMappingEigen<double> classifier(km, threshold, labels_map);

	try {
		classifier.classify(mnist, 10);
	}
	catch (const std::out_of_range& e) {
		std::cerr << "Classification error: " << e.what() << std::endl;
		throw;
	}
	catch (const std::exception& e) {
		std::cerr << "Unexpected error: " << e.what() << std::endl;
		throw;
	}

	// ----------------------
	// Plotting the results
	// ----------------------
	Plotter plotter;
	classifier.plot(plotter);
	for (int i = 0; i < 10 * 10; i += 9) {
		plotter.context().set_title("Mapping weights").show_heatmap(km.get_weights(i).data(), 28, 28, "gray");
	}
	plotter.wait(); 

	// We now explore the ways in which we can exploit our topologically meaningful
	// mapping to gain informations about the clusters in the original dataset. 
	// We present three such possible approaches: two variations of the classical U-Matrix
	// method, and k-means clustering performed on the unit weights. 
	Cluster<double> distance_u_matrix(km);
	KMeans kmeans_on_units(10, /* Iterations */ 50);
	UClusteringEigen<double> u_clustering(km);

	distance_u_matrix.compute();
	u_clustering.compute();
	kmeans_on_units.fit(km);

	// Plot all results and analyze what we have achieved: 
	distance_u_matrix.plot(plotter);
	u_clustering.plot(plotter);
	kmeans_on_units.plot(plotter);

}