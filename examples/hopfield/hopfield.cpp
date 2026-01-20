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

#include "io/datasets/dataset.hpp"
#include "io/datasets/repository.hpp"

int main() {
	
	// Create a plotter object
	Plotter p;

	using Hebbs = HebbianPolicy<float>;

	DenseHopfieldNetwork<Hebbs> dhn(40 * 40);
	StochasticHopfieldNetwork<Hebbs> shn(40 * 40);

	BinaryState bs1(40 * 40), bs2(40 * 40), bs3(40 * 40);

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
	logger.set_recording_stream(save_file, true);

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
	dhn.run(200, uc);
	dhn.detach_logger(&logger);

	HopfieldClassifier classifier;
	classifier.put_mapping(bs1_orig, 1);
	classifier.put_mapping(bs2, 2);

	dhn.attach_classifier(&classifier);
	auto cls = dhn.classify();
	if (cls)
		std::cout << "Classificazione: " << std::get<1>(*(dhn.classify()));

	HebbianCrossTalkTermVisualizer<float> cttv(p, 40 * 40);
	std::cout << "Devo calcola" << std::endl;

	cttv.compute_cross_talk_view(bs1, { &bs2, &bs3 });
	std::cout << "Devo showa" << std::endl;
	cttv.show(40, 40);


	p.block();



}