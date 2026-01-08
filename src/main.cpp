#include <cstdio>
#include <memory>
#include <thread>
#include <iostream>

// Hopfield networks
#include "hopfield/states/binary.hpp"
#include "hopfield/deterministic/dense_hopfield_network.hpp"
#include "hopfield/stochastic/stochastic_hopfield_network.hpp"
#include "hopfield/deterministic/cross_talk_visualizer.hpp"
#include "hopfield/logger/logger.hpp"

// Reservoir computing
#include "math/autograd/variables.hpp"
#include "reservoir/reservoir.hpp"
#include "reservoir/reservoir_predictor.hpp"
#include "reservoir/reservoir_logger.hpp"

// Konohen mappings
#include "mappings/konohen_mapping.hpp"
#include "mappings/classifier/majority_mapping.hpp"

// Restricted Boltzmann machines
#include "boltzmann/restricted_boltzmann_machine.hpp"
#include "boltzmann/boltzmann_logger.hpp"

// Io routines
#include "io/plot/plot.hpp"
#include "io/gif/gif.hpp"
#include "io/image/images.hpp"
#include "io/datasets/dataset.hpp"
#include "io/datasets/repository.hpp"

#include "utils/timing.hpp"

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

	HebbianCrossTalkTermVisualizer cttv(p, 40*40);
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
	DatasetRepo::load_mit_bih( "nowhere", 100, ecg_series);

	VectorDataset<Eigen::VectorXf, Eigen::VectorXf> sine_series(100);
	DatasetRepo::load_sine_time_series<float>(/* amount */300, sine_series);
	
	NetworkTrainer<float> trainer ( mlp ) ;
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

	RestrictedBoltzmannMachine<float> machine(64*64, 150);
	machine.initialize_weights(0.01);

	VectorCollection<VectorXf>  dataset(1000);
	DatasetRepo::load_real_faces_128("archive_reduced", 1000, dataset);
	for (int i = 0; i < 5; ++i)
	{
		p.context().show_heatmap(dataset.x_of(i).data(), 64, 64, "greyscale");
	}

	std::cout << "Loaded the dataset! " <<  dataset.size() << std::endl;
	SegmentTimer timer; 
	machine.load_weights("weights.pt");
	{
		auto scoped = timer.scoped("Training");

		machine.train_cd(20, dataset, 1, 0.04, 10);
	}
	for (int i = 0; i < 8; ++i) {
		machine.plot_kernel(p, i, 64, 64);

		machine.random_visible();
		machine.run_cd_k(60, 1.5, true);
		machine.plot_state(p, 64, 64);
	}

	machine.dump_weights("weights.pt");

	// timer.print();
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

void 
test_1D_neighbour_func() {
	// Parameters
	double sigma0 = 2.0;
	double tau = 10.0;
	unsigned int map_width = 10;
	unsigned int map_height = 1;  // 1D map
	std::string evolving_func = "exponential";

	// Create NeighbouringFunction
	NeighbouringFunction nf(sigma0, tau, map_width, map_height, evolving_func);

	// Set support size
	nf.set_support_size(5);  // 2 neighbors on each side

	// Pick a winner neuron
	unsigned int winner = 2;

	std::cout << "Winner neuron: " << winner << "\n";
	std::cout << "Neighborhood indices and weights:\n";

	// Iterate over 1D neighborhood
	for (auto it = nf.begin(winner); it != nf.end(winner); ++it) {
		std::cout << "x: " << it.index()
			<< ", weight: " << it.contribution_weight() << "\n";
	}
}

void
test_2D_neighbour_func() {
		// Parameters
		double sigma0 = 2;
		double tau = 10.0;
		unsigned int map_width = 10;
		unsigned int map_height = 4;  // 1D map
		std::string evolving_func = "exponential";

		// Create NeighbouringFunction
		NeighbouringFunction nf(sigma0, tau, map_width, map_height, evolving_func);

		// Set support size
		nf.set_support_size(3);  // 2 neighbors on each side

		// Pick a winner neuron
		unsigned int winner_x = 8;
		unsigned int winner_y = 3;

		std::cout << "Winner neuron: (" << winner_x << ", " << winner_y << ")" << "\n";
		std::cout << "Neighborhood indices and weights:\n";

		std::cout << "end neuron: (" << nf.end(winner_x, winner_y).x_coordinate() << ", " << nf.end(winner_x, winner_y).y_coordinate() << ")"
			<< "\n";

		int current_y = 0;
		int counter = 0;

		// Iterate over 2D neighborhood
		for (auto it = nf.begin(winner_x, winner_y); it != nf.end(winner_x, winner_y); ++it) {
			counter++;
			if (it.y_coordinate() != current_y) {
				current_y = it.y_coordinate();
				std::cout << "\n";
			}

			std::cout << "(" << it.x_coordinate() << ", " << it.y_coordinate() << ")" 
				<< ",: " << static_cast<double>(it.contribution_weight()) << "     ";
			
			if (counter == 40) {
				//break;
			}
			
			
		}
}

void 
test_1D_konohen_map() {
	// ----------------------
  // Parameters
  // ----------------------
	unsigned int input_size = 3;       // each input vector has 3 features
	unsigned int map_size = 5;         // 1D map with 5 neurons
	unsigned int iterations = 100;
	double learning_rate = 0.5;

	// ----------------------
	// Create map and neighborhood
	// ----------------------
	KonohenMap<double> km(map_size, input_size);
	
	// Parameters
	double sigma0 = 2.0;
	double tau = 10.0;
	unsigned int map_width = 5;
	unsigned int map_height = 1;  // 1D map
	std::string evolving_func = "exponential";

	// Create NeighbouringFunction
	NeighbouringFunction nf(sigma0, tau, map_width, map_height, evolving_func);

	// Set support size
	nf.set_support_size(3);  // 2 neighbors on each side

	// ----------------------
	// Initialize weights
	// ----------------------
	km.initialize(42);

	// Print initial weights
	std::cout << "Initial weights:\n";
	for (unsigned int i = 0; i < map_size; ++i) {
		std::cout << "Neuron " << i << ": ";
		for (unsigned int j = 0; j < input_size; ++j) {
			std::cout << km.get_weights(i)[j] << " ";
		}
		std::cout << "\n";
	}

	// ----------------------
	// Create training data (3 vectors of 3 elements)
	// ----------------------
	std::vector<std::unique_ptr<double[]>> data;
	for (int i = 0; i < 3; ++i) {
		auto vec = std::make_unique<double[]>(input_size);
		for (int j = 0; j < input_size; ++j) {
			vec[j] = i + 0.1 * j;  // simple values: 0, 1.1, 2.2 etc
		}
		data.push_back(std::move(vec));
	}

	std::cout << "\nInput vectors:\n";
	for (unsigned int i = 0; i < 3; ++i) {
		std::cout << "Vector " << i << ": ";
		for (unsigned int j = 0; j < input_size; ++j) {
			std::cout << data[i][j] << " ";
		}
		std::cout << "\n";
	}

	// ----------------------
	// Train the map
	// ----------------------
	km.train(data, iterations, nf, learning_rate);

	// ----------------------
	// Display resulting weights
	// ----------------------
	std::cout << "\nTrained weights:\n";
	for (unsigned int i = 0; i < map_size; ++i) {
		std::cout << "Neuron " << i << ": ";
		for (unsigned int j = 0; j < input_size; ++j) {
			std::cout << km.get_weights(i)[j] << " ";
		}
		std::cout << "\n";
	}
}

void
test_2D_konohen_map() {
	// ----------------------
  // Parameters
  // ----------------------
	unsigned int input_size = 3;       // each input vector has 3 features
	unsigned int iterations = 100;
	double learning_rate = 0.5;


	// Parameters
	double sigma0 = 2.0;
	double tau = 10.0;
	unsigned int map_width = 5;
	unsigned int map_height = 3; 
	std::string evolving_func = "exponential";

	// ----------------------
	// Create map and neighborhood
	// ----------------------
	KonohenMap<double> km(map_width, map_height, input_size);

	// Create NeighbouringFunction
	NeighbouringFunction nf(sigma0, tau, map_width, map_height, evolving_func);

	// Set support size
	nf.set_support_size(1);  // 2 neighbors on each side

	// ----------------------
	// Initialize weights
	// ----------------------
	km.initialize(42);

	// Print initial weights
	std::cout << "Initial weights:\n";
	for (unsigned int k = 0; k < map_height; ++k) {
		for (unsigned int i = 0; i < map_width; ++i) {
			std::cout << "index " << i + k * map_width << "\n ";
			std::cout << "Neuron (" << i << ", "<< k <<")" << ": ";
			for (unsigned int j = 0; j < input_size; ++j) {
				std::cout << km.get_weights(i + k*map_width)[j] << " ";
			}
			std::cout << "\n";
		}
		
	}

	// ----------------------
	// Create training data (3 vectors of 3 elements)
	// ----------------------
	std::vector<std::unique_ptr<double[]>> data;
	for (int i = 0; i < 5; ++i) {
		auto vec = std::make_unique<double[]>(input_size);
		for (int j = 0; j < input_size; ++j) {
			vec[j] = i + 0.1 * j;  // simple values: 0, 1.1, 2.2 etc
		}
		data.push_back(std::move(vec));
	}

	std::cout << "\nInput vectors:\n";
	for (unsigned int i = 0; i < 5; ++i) {
		std::cout << "Vector " << i << ": ";
		for (unsigned int j = 0; j < input_size; ++j) {
			std::cout << data[i][j] << " ";
		}
		std::cout << "\n";
	}

	// ----------------------
	// Train the map
	// ----------------------
	km.train(data, iterations, nf, learning_rate);

	// ----------------------
	// Display resulting weights
	// ----------------------
	std::cout << "\nTrained weights:\n";
	for (unsigned int k = 0; k < map_height; ++k) {
		for (unsigned int i = 0; i < map_width; ++i) {
			std::cout << "index " << i + k * map_width << "\n ";
			std::cout << "Neuron (" << i << ", " << k << ")" << ": ";
			for (unsigned int j = 0; j < input_size; ++j) {
				std::cout << km.get_weights(i + k * map_width)[j] << " ";
			}
			std::cout << "\n";
		}

	}
}

void
cluster_classification_2D() {
	// ----------------------
  // Parameters
  // ----------------------
	unsigned int input_size = 2;       // each input vector has 3 features
	unsigned int iterations = 300;
	double learning_rate = 0.5;


	// Parameters
	double sigma0 = 2.0;
	double tau = 10.0;
	unsigned int map_width = 10;
	unsigned int map_height = 2;
	std::string evolving_func = "exponential";

	// ----------------------
	// Create map and neighborhood
	// ----------------------
	KonohenMap<double> km(map_width, map_height, input_size);

	// Create NeighbouringFunction
	NeighbouringFunction nf(sigma0, tau, map_width, map_height, evolving_func);

	// Set support size
	nf.set_support_size(2);  // 2 neighbors on each side

	// ----------------------
	// Initialize weights
	// ----------------------
	km.initialize(42);

	// Print initial weights
	std::cout << "Initial weights:\n";
	for (unsigned int k = 0; k < map_height; ++k) {
		for (unsigned int i = 0; i < map_width; ++i) {
			std::cout << "Neuron (" << i << ", " << k << ")" << ": ";
			for (unsigned int j = 0; j < input_size; ++j) {
				std::cout << km.get_weights(i + k * map_width)[j] << " ";
			}
			std::cout << "\n";
		}

	}

	// ----------------------
	// Create training data (3 vectors of 3 elements)
	// ----------------------
	std::vector<std::unique_ptr<double[]>> data;
	for (int i = 0; i < 5; ++i) {
		auto vec = std::make_unique<double[]>(input_size);
		for (int j = 0; j < input_size; ++j) {
			vec[j] = 1 + i*0.1;  // simple values: 0, 1.1, 2.2 etc
		}
		data.push_back(std::move(vec));
	}

	for (int i = 5; i < 10; ++i) {
		auto vec = std::make_unique<double[]>(input_size);
		for (int j = 0; j < input_size; ++j) {
			vec[j] = 55 + i*0.1;  // simple values: 0, 1.1, 2.2 etc
		}
		data.push_back(std::move(vec));
	}

	std::cout << "\nInput vectors:\n";
	for (unsigned int i = 0; i < 10; ++i) {
		std::cout << "Vector " << i << ": ";
		for (unsigned int j = 0; j < input_size; ++j) {
			std::cout << data[i][j] << " ";
		}
		std::cout << "\n";
	}

	// ----------------------
	// Train the map
	// ----------------------
	km.train(data, iterations, nf, learning_rate);

	// ----------------------
	// Display resulting weights
	// ----------------------
	std::cout << "\nTrained weights:\n";
	for (unsigned int k = 0; k < map_height; ++k) {
		for (unsigned int i = 0; i < map_width; ++i) {
			std::cout << "Neuron (" << i << ", " << k << ")" << ": ";
			for (unsigned int j = 0; j < input_size; ++j) {
				std::cout << km.get_weights(i + k * map_width)[j] << " ";
			}
			std::cout << "\n";
		}

	}
}

void classification_test() {

  // Parameters
	unsigned int input_size = 2;       // each input vector has 3 features
	unsigned int iterations = 300;
	double learning_rate = 0.5;


	// Parameters
	double sigma0 = 2.0;
	double tau = 10.0;
	unsigned int map_width = 10;
	unsigned int map_height = 2;
	std::string evolving_func = "exponential";


	KonohenMap<double> km(map_width, map_height, input_size);

	// Create NeighbouringFunction
	NeighbouringFunction nf(sigma0, tau, map_width, map_height, evolving_func);

	// Set support size
	nf.set_support_size(2);  

	km.initialize(42);

	std::vector<std::unique_ptr<double[]>> data;
	for (int i = 0; i < 5; ++i) {
		auto vec = std::make_unique<double[]>(input_size);
		for (int j = 0; j < input_size; ++j) {
			vec[j] = 1 + i * 0.1;  // simple values: 0, 1.1, 2.2 etc
		}
		data.push_back(std::move(vec));
	}

	for (int i = 5; i < 10; ++i) {
		auto vec = std::make_unique<double[]>(input_size);
		for (int j = 0; j < input_size; ++j) {
			vec[j] = 55 + i * 0.1;  // simple values: 0, 1.1, 2.2 etc
		}
		data.push_back(std::move(vec));
	}

	km.train(data, iterations, nf, learning_rate);

	std::map<int, std::string> labels_map;
	labels_map[0] = "ones";
	labels_map[1] = "fifties";

	MajorityMapping<double> classifier(km, 0.8, labels_map); 

	Dataset<double, int> dataset(10);

	for (int i = 0; i < 5; ++i) {
		dataset.add_sample(data[i].get(), 0);
	}

	for (int i = 5; i < 10; ++i) {
		dataset.add_sample(data[i].get(), 1);
	}

	classifier.classify(dataset);

	for (unsigned int k = 0; k < map_height; ++k) {
		for (unsigned int i = 0; i < map_width; ++i) {
			std::cout << "Neuron (" << i << ", " << k << ")" << ": ";
			for (unsigned int j = 0; j < input_size; ++j) {
				std::cout << km.get_weights(i + k * map_width)[j] << " ";
			}
			std::cout << "\n labeled as: ";
			std::cout << classifier.label_for(i+k*map_width) << "\n\n";
		}

	}

}

// Just create the folder...
int main() {

	std::cout << "Number of threads: " << std::thread::hardware_concurrency();
	Eigen::initParallel();
	Eigen::setNbThreads(std::thread::hardware_concurrency());

	std::cout << "Eigen threads: " << Eigen::nbThreads() << "\n";
	// classification_test();

	// autograd_compile();
	/*
	autograd_compile();
	io_utils_compile();
	*/



	boltzmann_compile();
	// reservoir_compile();
	// hopfield_compile();

}
	