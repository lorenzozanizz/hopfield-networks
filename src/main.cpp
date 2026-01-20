#include <cstdio>
#include <memory>
#include <thread>
#include <iostream>

// Hopfield networks
#include "hopfield/states/binary.hpp"
#include "hopfield/deterministic/dense_hopfield_network.hpp"
#include "hopfield/stochastic/stochastic_hopfield_network.hpp"
#include "hopfield/deterministic/cross_talk_visualizer.hpp"

#include "hopfield/parallelized/parallelized_deterministic.hpp"
#include "hopfield/parallelized/parallelized_stochastic.hpp"
#include "hopfield/logger/logger.hpp"

// Reservoir computing
#include "math/autograd/variables.hpp"
#include "reservoir/reservoir.hpp"
#include "reservoir/reservoir_predictor.hpp"
#include "reservoir/reservoir_logger.hpp"

// Konohen mappings
/*
#include "mappings/konohen_mapping.hpp"
#include "mappings/classifier/majority_mapping.hpp"
#include "mappings/clustering/u_clustering.hpp"
*/

#include "mappings/konohen_mapping_eigen.hpp"
#include "mappings/classifier/majority_mapping_eigen.hpp"
#include "mappings/clustering/u_clustering_eigen.hpp"

// Restricted Boltzmann machines
#include "boltzmann/restricted_boltzmann_machine.hpp"
#include "boltzmann/deep_belief_network.hpp"
#include "boltzmann/boltzmann_logger.hpp"

// Io routines
#include "io/plot/plot.hpp"
#include "io/gif/gif.hpp"
#include "io/image/images.hpp"
#include "io/datasets/dataset.hpp"
#include "io/datasets/dataset_eigen.hpp"
#include "io/datasets/repository.hpp"

#include <cfenv> 
#pragma STDC FENV_ACCESS ON

enum NeighbouringStrategy {
	OneDNeighbouring,
	TwoDNeighbouring,
	ThreeDNeighbouring
};


void
autograd_compile() {

	using namespace autograd;

	ScalarFunction func; // a scalar function
	auto& g = func.generator();
	auto u = g.create_vector_variable(30);
	const double lambda = 0.01;
	const auto expr = g.sum(
		g.squared_l2_norm((g.sub(u, 1.0))),
		g.multiply(lambda, g.squared_l2_norm(u))
	);
	func = expr;

	EvalMap<float> map;
	EigVec<float> vec(30);
	vec.setZero();


	map.emplace(u, std::reference_wrapper(vec));
	std::cout << func;

	std::cout << "FINAL VALUE IS  = " << func(map) << std::endl;

	VectorFunction deriv;
	func.derivative(deriv, u);
	std::cout << deriv;
	EigVec<float> res(30);

	deriv(map, res);
		
}


void
hopfield_compile() {
	
	Plotter p;

	DenseHopfieldNetwork<HebbianPolicy<float>> dhn(40 * 40);
	StochasticHopfieldNetwork<HebbianPolicy<float>> shn(40 * 40);

	BinaryState bs1(40 * 40), bs2(40*40), bs3(40*40);
	auto img1 = "flower.png";
	auto img2 = "car.png";
	auto img3 = "kitty.png";

	Image img1_r(img1, Channels::Greyscale);
	{ p.context().show_image(img1_r); }

	VectorDataset<std::vector<unsigned char>, unsigned int> mnist(100);
	DatasetRepo::load_mnist_vector("C:/Users/picul/Documents/MNIST/vector_mnist.data", 100, mnist);

	StateUtils::load_state_from_image(bs1, img1_r,  /* binarize */ true);

	{ p.context().show_image(img1_r); }

	bs1.set_stride_y(40); // mark the state as 2 dimensional, 40 x 40
	StateUtils::plot_state(p, bs1);

	std::cout << "Loaded image!" << std::endl;
	dhn.store(bs1);

	StateUtils::load_state_from_image(bs2, img2, /* binarize */ true);
	

	StateUtils::load_state_from_image(bs3, img3, /* binarize */ true);
	dhn.store(bs3);

	auto& weighting_policy = dhn.weighting_policy();

	// Create and open a text file 
	auto logger = HopfieldLogger(&p);
	std::ofstream save_file("save_data.txt");
	logger.set_recording_stream(save_file,  true);

	dhn.attach_logger(&logger);
	logger.set_collect_states(true, "states.gif");
	logger.set_collect_energy(true);
	logger.set_collect_temperature(true);
	logger.set_collect_order_parameter(true);

	logger.finally_write_last_state_png(true, "last_state.png");
	logger.finally_plot_data(true);

	BinaryState bs1_orig(40 * 40);
	bs1_orig.copy_content(bs1);

	StateUtils::perturb_state(bs1,  /* Noise intensity 0 to 1*/ 0.3);
	std::cout << "Plotting state!" << std::endl;
	StateUtils::plot_state(p, bs1);

	UpdateConfig uc = {
		UpdatePolicy::GroupUpdate,
		/* group size */ 60
	};


	// Instruct the network that we wish to interpret the state as a 40x40 raster.
	dhn.set_state_strides(40);
	dhn.add_reference_state(bs1_orig);
	dhn.add_reference_state(bs2);
	dhn.add_reference_state(bs3);

	dhn.feed(bs1);
	dhn.run( 200, uc);
	dhn.detach_logger(&logger);

	HopfieldClassifier classifier;
	classifier.put_mapping(bs1_orig, 1);
	classifier.put_mapping(bs2, 2);

	dhn.attach_classifier(&classifier);
	auto cls = dhn.classify();
	if (cls)
		std::cout << "Classificazione: " << std::get<1>(*(dhn.classify()));

	HebbianCrossTalkTermVisualizer<float> cttv(p, 40*40);
	std::cout << "Devo calcola" << std::endl;

	cttv.compute_cross_talk_view(bs1, { &bs2, &bs3 });
	std::cout << "Devo showa" << std::endl;
	cttv.show(40, 40);


	p.block();
	
	
}

void 
konohen_compile() {

}

void 
reservoir_compile() {

	Reservoir<float> reservoir(10, 30);
	ReservoirLogger<float> logger;

	reservoir.attach_logger(&logger);


	reservoir.initialize_echo_weights(0.1, SamplingType::Uniform);
	reservoir.initialize_input_weights(SamplingType::Normal);


	Plotter p;
	logger.set_collect_norm(true);
	logger.set_collect_states(true, "res_states.gif", 6, 6);
	// logger.assign_plotter(&p);

	Eigen::VectorXf input(10);
	input(1) = 0.4;
	input(2) = 0.5;
	input(0) = 0.1;

	reservoir.begin_run();
	reservoir.feed(input);
	for (int i = 0; i < 20; ++i) {
		reservoir.run(); 
	}
	reservoir.end_run();


	// feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);

	std::vector<int> units = { 30, 30, 10 };
	auto sigmoid = Activations<float>::sigmoid;
	auto identity = Activations<float>::identity;

	std::vector<ActivationFunction<float>> acts = { sigmoid , sigmoid };

	MultiLayerPerceptron<float> mlp(units, acts) ;
	
	Eigen::MatrixXf inputs(30, 4); 


	using namespace autograd;

	ScalarFunction loss_function;
	auto& g = loss_function.generator();
	auto y		= g.create_vector_variable(10);
	auto y_hat  = g.create_vector_variable(10);
	loss_function = g.squared_l2_norm(g.sub(y, y_hat));

	EvalMap<float> map;
	EigVec<float> vec(10);
	vec.setZero();

	EigVec<float> vec_ref(10);
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> batch_ref(10, 2);
	vec_ref.setZero();
	vec_ref(2) = 0.5;
	vec_ref(1) = 0.5;

	batch_ref.setZero();
	batch_ref(2, 0) = 0.5;
	batch_ref(1, 0) = 0.5;
	batch_ref(3, 1) = 0.8;
	batch_ref(1, 1) = 0.2;

	EigVec<float> vec_input(30);
	vec_input(10) = 3.0;

	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> batch_input(30, 2);
	batch_input(1, 1) = 1.0;
	batch_input(10, 0) = 3.0;

	// Eigen::MatrixXf out = mlp.forward(batch_input);
	
	// vec = out.col(0);

	// map.emplace(y, vec);
	// map.emplace(y_hat, vec_ref);

	// std::cout << "Value of the loss : " << loss_function( map ) << std::endl;

	VectorFunction deriv;
	loss_function.derivative(deriv, y);
	
	// std::cout << loss_function;

	// std::cout << deriv;

	// Eigen::VectorXf loss_grad(10);
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> batch_grad(10, 2);

	Eigen::MatrixXf out = mlp.forward(batch_input);

	EigVec<float> out_col = out.col(0);
	EigVec<float> true_col = batch_ref.col(0);
	map.clear();

	map.emplace(y, out_col);
	map.emplace(y_hat, true_col);

	auto loss = loss_function(map);
	std::cout << "LOSS INITIALLY: " << loss;

	/*
	for (int i = 0; i < 20; ++i) {
		
		Eigen::MatrixXf out = mlp.forward(batch_input);

		for (int batch = 0; batch < 2; ++batch) {
			// auto grad_col = batch_grad.col(batch); 
			// auto out_col = out.col(batch);
			EigVec<float> out_col = out.col(batch);
			EigVec<float> true_col = batch_ref.col(batch);
			map.clear();

			map.emplace(y, out_col);
			map.emplace(y_hat, true_col);
			deriv(map, batch_grad.col(batch));
		}

		mlp.backward(batch_grad);
		mlp.apply_gradients(0.2);
	}
	*/

	VectorDataset<Eigen::VectorXf, Eigen::VectorXf> ecg_series(100);
	DatasetRepo::load_mit_bih("mitbih_combined_records.csv", 100, ecg_series);

	VectorDataset<Eigen::VectorXf, Eigen::VectorXf> sine_series(100);
	DatasetRepo::load_sine_time_series<float>(/* amount */300, sine_series);
	
	NetworkTrainer<float, MultiLayerPerceptron<float>> trainer ( mlp ) ;
	trainer.set_loss_function(& loss_function, y, y_hat);

	trainer.train(
		100,
		sine_series,
		std::nullopt, // no verification dataset
		5,
		0.04 //alpha
	);

	out = mlp.forward(batch_input);

	out_col = out.col(0);
	true_col = batch_ref.col(0);
	map.clear();

	map.emplace(y, out_col);
	map.emplace(y_hat, true_col);

	loss = loss_function(map);
	std::cout << "LOSS finally: " << loss;


}

void 
boltzmann_compile() {

	using namespace Eigen;

	Plotter p;

	RestrictedBoltzmannMachine<float> machine(28*28, 529);
	RestrictedBoltzmannMachine<float> machine2(529, 361);

	machine.initialize_weights(0.008);
	machine2.initialize_weights(0.008);

	VectorCollection<VectorXf>  dataset(800);
	DatasetRepo::load_mnist_eigen<float>("vector_mnist.data", 800, dataset,
		/*threshold */ 127, 0.0, 1.0);

	std::cout << "Loaded the dataset!" <<  dataset.size() << std::endl;

	machine.train_cd(50, dataset, 25, 0.02, 10);
	
	VectorCollection<VectorXf>  out_dataset(800);
	machine.map_into_hidden(dataset, out_dataset, 25);

	p.context().set_title("State").show_heatmap(out_dataset.x_of(0).data(), 23, 23, "gray");
	for (int i = 0; i < 8; ++i) {
		machine.random_visible();
		machine.plot_kernel(p, i, 28, 28);
	}

	p.block();

	machine2.train_cd(40, out_dataset, 25, 0.02, 10);
	for (int i = 0; i < 8; ++i) {
		machine2.random_visible();
		machine2.plot_kernel(p, i, 23, 23);
		machine2.plot_higher_order_kernel(p, i, 28, 28, machine);
	}

	p.block();


}

void
io_utils_compile() {

	
	// GnuplotPipe gp;
	// gp.send_line("plot [-pi/2:pi] cos(x),-(sin(x) > sin(x+1) ? sin(x) : sin(x+1))");

	unsigned char* data = new unsigned char[50 * 50 * 4];
	memset(data, 0, 50 * 50 * 4);
	GifWriterIO gio;
	{
		auto context = gio.initialize_writing_context("duddu.gif", 50, 50, 100);
		context.write(data);
		memset(data, 255, 50 * 50 * 4);
		context.write(data);
		memset(data, 0, 50 * 50 * 2);
		context.write(data);
	}

	Image img("bilbo.png", Channels::Greyscale);

}

using DoubleVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
void classification_MNIST() {
	// ----------------------
	// Loading MNIST
	// ----------------------
	VectorDataset<DoubleVector, unsigned int> mnist(400);
	DatasetRepo::load_mnist_eigen("vector_mnist.data", 400, mnist);
	
	// #NOTE: personalmente a me confonde un po' tutta sta roba, ma visto che la mappa sembra
	// funzionare teniamo così e via, documenta però bene i costruttori della evolving func
	// e in caso alcune configurazioni (tipo lineare) non usano alcuni parametri crea multipli
	// costruttori
	// 
	// ----------------------
	// Setting the parameters
	// ----------------------
	// 
	// Map parameters
	unsigned int input_size = 28 * 28; 
	unsigned int map_width = 10; 
	unsigned int map_height = 10;
	unsigned int iterations = 100; // number of iterations to perform in the training of the Kohonen map
	double learning_rate = 0.10; // weight of the update in each iteration in the Kohonen map

	// Evolving function parameters (how the sigma changes through the iterations)
	std::string evolving_func = "exponential"; // Other options: linear, piecewise and inverse_time 
	double sigma0 = 3.0; 
	double tau = 10.0;
	double sigma1 = 1.0; // used in the piecewise evolving function
	double beta = 0.0; // used in the inverse_time method

	// Majority map parameters
	double threshold = 0.6; // the percentage of hits required to label a neuron in the classifier
	

	// ----------------------
	// Create map and neighborhood
	// ----------------------
	KonohenMapEigen<double> km(map_width, map_height, input_size);
	NeighbouringFunctionEigen nf(sigma0, tau, map_width, map_height, evolving_func);
	nf.set_sigma_1(sigma1);
	nf.set_beta(beta);
	nf.set_t_max(iterations);

	// ----------------------
	// Initializing and training Konohen map
	// ----------------------
	nf.set_support_size(2);
	km.initialize();
	km.train(mnist, iterations, nf, learning_rate);

	// ----------------------
	// Creating the label map for MNIST
	// ----------------------
	std::map<int, std::string> labels_map;
	labels_map[0] = "unknown";
	labels_map[1] = "zero";
	labels_map[2] = "one";
	labels_map[3] = "two";
	labels_map[4] = "three";
	labels_map[5] = "four";
	labels_map[6] = "five";
	labels_map[7] = "six";
	labels_map[8] = "seven";
	labels_map[9] = "eight";
	labels_map[10] = "nine";
	
	// ----------------------
	// Initializing the majority map and performing the classification
	// ----------------------
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
		plotter.context().show_heatmap(km.get_weights(i).data(), 28, 28, "gray");
	}
	plotter.block(); // NOTE: this is to move in the function that calls this, is here to remember 
	// #note: crea una funzione che visualizza i kernel in questo modo, la mettiamo negli
	// examples. 

}

using DoubleVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
void clustering_MNIST() {

	// ----------------------
	// Loading MNIST
	// ----------------------
	VectorDataset<DoubleVector, unsigned int> mnist(80); 
	DatasetRepo::load_mnist_eigen("vector_mnist.data", 80, mnist);

	// Map parameters
	unsigned int input_size = 28 * 28;
	unsigned int map_width = 6;
	unsigned int map_height = 6;
	unsigned int iterations = 200; // number of iterations to perform in the training of the Kohonen map
	double learning_rate = 0.30; // weight of the update in each iteration in the Kohonen map

	// Evolving function parameters (how the sigma changes through the iterations)
	std::string evolving_func = "exponential"; // Other options: linear, piecewise and inverse_time 
	double sigma0 = 3.0;
	double tau = 10.0;
	double sigma1 = 1.0; // used in the piecewise evolving function
	double beta = 0.0; // used in the inverse_time method

	// ----------------------
	// Create map and neighborhood
	// ----------------------
	KonohenMapEigen<double> km(map_width, map_height, input_size);
	NeighbouringFunctionEigen nf(sigma0, tau, map_width, map_height, evolving_func);

	// ----------------------
	// Initializing and training Konohen map
	// ----------------------
	nf.set_support_size(2);
	km.initialize();
	km.train(mnist, iterations, nf, learning_rate);

	// ----------------------
	// Initializing the UMatrix and performing the clustering
	// ----------------------
	UClusteringEigen<double> UMap(km);
	UMap.compute();

	// ----------------------
	// Plotting the results
	// ----------------------
	Plotter plotter;
	UMap.plot(plotter);

}

using DoubleVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
void clustering() {

	// ----------------------
	// Loading MNIST
	// ----------------------
	unsigned int size = 30;
	unsigned int input_size = 4;
	VectorDataset<DoubleVector, unsigned int> mnist(size);
	for (int i=0; i<size/4; ++i) {
		Eigen::Matrix<double, Eigen::Dynamic, 1> v(4);
		v << 1.0, 0.0, 0.0, 4.0;
		mnist.add_sample(v, 0, i);
	}
	for (int i = size / 4; i < size * 2 / 4; ++i) {
		Eigen::Matrix<double, Eigen::Dynamic, 1> v(4);
		v << 0.0, 46.0, 1.0, 4.0;
		mnist.add_sample(v, 1, i);
	}
	for (int i = size * 2 / 4; i < size * 3 / 4; ++i) {
		Eigen::Matrix<double, Eigen::Dynamic, 1> v(4);
		v << 0.0, 0.0, 11.0, 0.0;
		mnist.add_sample(v, 2, i);
	}
	for (int i = size * 3 / 4; i < size ; ++i) {
		Eigen::Matrix<double, Eigen::Dynamic, 1> v(4);
		v << 33.0, 78.0, 11.0, 0.0;
		mnist.add_sample(v, 3, i);
	}
	// Map parameters
	
	unsigned int map_width = 5;
	unsigned int map_height = 5;
	unsigned int iterations = 200; // number of iterations to perform in the training of the Kohonen map
	double learning_rate = 0.3; // weight of the update in each iteration in the Kohonen map

	// Evolving function parameters (how the sigma changes through the iterations)
	std::string evolving_func = "exponential"; // Other options: linear, piecewise and inverse_time 
	double sigma0 = 3.0;
	double tau = 10.0;
	double sigma1 = 1.0; // used in the piecewise evolving function
	double beta = 0.0; // used in the inverse_time method

	// ----------------------
	// Create map and neighborhood
	// ----------------------
	KonohenMapEigen<double> km(map_width, map_height, input_size);
	NeighbouringFunctionEigen nf(sigma0, tau, map_width, map_height, evolving_func);

	// ----------------------
	// Initializing and training Konohen map
	// ----------------------
	nf.set_support_size(2);
	km.initialize();
	km.train(mnist, iterations, nf, learning_rate);


	for (int i = 0; i < map_width; ++i) {
		for (int j = 0; j < map_height; ++j) {
			std::cout <<"( " << i <<"," <<j <<" ): " << km.get_weights(i, j) << "\n";
		}
		std::cout <<  "\n\n";
	}

	// ----------------------
	// Initializing the UMatrix and performing the clustering
	// ----------------------
	UClusteringEigen<double> UMap(km);
	UMap.compute();

	// ----------------------
	// Plotting the results
	// ----------------------
	Plotter plotter;
	UMap.plot(plotter);

}

// Just create the folder...
int main() {
	//test_2D_konohen_map_Eigen();
	//classification_test_Eigen();
	//clustering_MNIST();
	//classification_MNIST();
	//clustering_test_eigen();

	Utilities::eigen_init_parallel( -1 );

	// clustering_MNIST(); 
	//classification_MNIST();

	// classification_test();

	//boltzmann_compile();
	// autograd_compile();
	/*
	autograd_compile();
	io_utils_compile();
	*/
	//reservoir_compile();
	// hopfield_compile();

}
	