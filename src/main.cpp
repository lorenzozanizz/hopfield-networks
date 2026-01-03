#include <cstdio>
#include <memory>
#include <iostream>

// Hopfield networks

#include "hopfield/states/binary.hpp"
#include "hopfield/deterministic/dense_hopfield_network.hpp"
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

#include "io/plot/plot.hpp"
#include "io/gif/gif.hpp"
#include "io/image/images.hpp"
#include "io/datasets/dataset.hpp"



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
	auto u = g.create_vector_variable(140);
	const double lambda = 0.01;
	const auto expr = g.sum(
		g.squared_norm((g.sub(u, 1.0))),
		g.multiply(lambda, g.squared_norm(u))
	);
	func = expr;

	EvalMap<float> map;
	EigVec<float> vec(140);
	vec.setZero();
	vec(0) = 1.0;
	vec(4) = 3.0;
	vec(100) = 6.0;
	vec(55) = 6.0;
	vec(88) = 6.0;

	// map.emplace(u, std::reference_wrapper(vec));
	// std::cout << func;

	// std::cout << "VALUE = " << func(map);

	// VectorFunction deriv;
	// func.derivative(deriv, u);

	// std::cout << deriv;
}


void
hopfield_compile() {
	
	Plotter p;

	DenseHopfieldNetwork<HebbianPolicy> dhn(40 * 40);

	BinaryState bs1(40 * 40), bs2(40*40), bs3(40*40);
	auto img1 = "flower.png";
	auto img2 = "car.png";
	auto img3 = "kitty.png";

	Image img1_r(img1, Channels::Greyscale);
	{ p.context().show_image(img1_r); }


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

	StateUtils::perturb_state(bs1,  /* Noise intensity 0 to 1*/ 0.15);
	std::cout << "Plotting state!" << std::endl;
	StateUtils::plot_state(p, bs1);

	UpdateConfig uc = {
		UpdatePolicy::Asynchronous
	};


	// Instruct the network that we wish to interpret the state as a 40x40 raster.
	dhn.set_state_strides(40);
	dhn.run(bs1, 20, uc);
	dhn.detach_logger(&logger);

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

}

void 
boltzmann_compile() {

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

	classification_test();

	/*
	autograd_compile();
	io_utils_compile();
	hopfield_compile();
	*/
	
}
	