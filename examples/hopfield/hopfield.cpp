#include <cstdio>
#include <memory>
#include <thread>
#include <iostream>

// -------------- HOPFIELD NETWORKS --------------
#include "hopfield/states/binary.hpp"
#include "hopfield/deterministic/dense_hopfield_network.hpp"
#include "hopfield/stochastic/stochastic_hopfield_network.hpp"
#include "hopfield/deterministic/cross_talk_visualizer.hpp"

#include "hopfield/parallelized/parallelized_deterministic.hpp"
#include "hopfield/parallelized/parallelized_stochastic.hpp"
#include "hopfield/logger/logger.hpp"

// -------------- INPUT ROUTINES --------------
#include "io/datasets/dataset.hpp"
#include "io/datasets/repository.hpp"
#include "io/image/images.hpp"

int main() {
	
	using Hebbs = HebbianPolicy<float>;
	using Storkey = StorkeyPolicy<float>;

	// Create a plotter object to plot what we require. 
	Plotter p;

	constexpr const auto MNIST_SIZE = 28;

	DenseHopfieldNetwork<Hebbs>			dhn(MNIST_SIZE * MNIST_SIZE);
	StochasticHopfieldNetwork<Storkey>	shn(MNIST_SIZE * MNIST_SIZE);

	VectorDataset<std::vector<unsigned char>, unsigned int> ten_representatives(10);
	// Load ten representatives for the ten classes of hopfield networks. 
	DatasetRepo::load_mnist_ten_categories("first_ten.data", ten_representatives);
	// We can plot some of these representatives (noting that they are loaded in order)

	std::cout << "> Loading the 10 patterns: " << std::endl;
	for (int i = 0; i < 3; ++i)
		p.context().show_image(
			ten_representatives.x_of(i).data(), MNIST_SIZE, MNIST_SIZE, 1);
			
	// Temporarily block the execution to allow visualizing the graphs
	// p.block();

	// We need to discretize the patterns to feed them into MNIST. 
	for (int i = 0; i < 10; ++i)
		ImageUtils::background_aware_binarize(
			ten_representatives.ref_of(i).data(), MNIST_SIZE, MNIST_SIZE);

	p.context().show_image(
		ten_representatives.x_of(0).data(), MNIST_SIZE, MNIST_SIZE, 1);
	// Show how this was discretized
	// p.block();

	// Load all the binary states with the image representatives. 
	std::vector<BinaryState> binary_states;
	binary_states.reserve(10);

	for (int i = 0; i < 10; ++i) {
		binary_states.emplace_back(MNIST_SIZE * MNIST_SIZE);
		// Mark the pattern as representing a 2d image. 
		binary_states[i].set_stride_y(MNIST_SIZE);
		StateUtils::load_state_from_byte_array(binary_states[i], ten_representatives.ref_of(i).data(), 
			MNIST_SIZE * MNIST_SIZE);
	}
	
	std::cout << "> Storing the patterns in the networks: " << std::endl;
	// Store all ten patterns in both networks
	for (int i = 0; i < 10; ++i) {
		dhn.store(binary_states[i]);
		shn.store(binary_states[i]);
	}

	std::cout << "> Initializing the logger: " << std::endl;
	// Create a logger object to visualize the action of the network over time. 
	auto logger = HopfieldLogger(&p);
	std::ofstream save_file("save_data.txt");
	{
		// Setup all configurations for the logger
		logger.set_recording_stream(save_file, true);

		logger.set_collect_states(true, "deterministic_states.gif");
		logger.set_collect_energy(true);
		logger.set_collect_temperature(true);

		// We collect order parameters, which represent averages in time
		// of the agreement between the state of the network and other patterns. 
		logger.set_collect_order_parameter(true);
		for (int i = 0; i < 10; ++i) {
			dhn.add_reference_state(binary_states[i]);
			shn.add_reference_state(binary_states[i]);
		}
		logger.set_prefix("Deterministic network | ");
		logger.finally_write_last_state_png(true, "deterministic_last_state.png");
		logger.finally_plot_data(true);
	}
	dhn.attach_logger(&logger);

	// --- Now display the denoising property of the hopfield network ---
	BinaryState deformed_5(MNIST_SIZE * MNIST_SIZE);
	deformed_5.copy_content(binary_states[5]);

	StateUtils::perturb_state(deformed_5, /* Noise level */ 0.15);

	dhn.set_state_strides(MNIST_SIZE);
	shn.set_state_strides(MNIST_SIZE);

	UpdateConfig uc = { UpdatePolicy::GroupUpdate, 60 };

	std::cout << "> Feeding the pattern to the network: " << std::endl;
	dhn.feed(deformed_5);
	std::cout << "> Running the deterministic network: " << std::endl;
	dhn.run(/* Iterations */ 200, uc);
	// Implicit plotting... because of the logger settings. 

	// The logger has generated a gif representing the transition of denoised states over
	// time "states.gif", has saved to memory the final state of the network "last_state.png"
	// and has collected debugging info for the order parameters and energy of the network. 
	dhn.detach_logger(&logger);

	// Now attach the logger to the stochastich hopfield network and see how it behaves.
	shn.attach_logger(&logger); {
		logger.set_collect_states(true, "stochastic_states.gif");
		logger.finally_write_last_state_png(true, "stochastic_last_state.png");
		logger.set_prefix("Stochastic network | ");
	}
	std::cout << "> Running the stochastic network: " << std::endl;

	deformed_5.copy_content(binary_states[5]);

	StateUtils::perturb_state(deformed_5, /* Noise level */ 0.15);
	std::unique_ptr<AnnealingScheduler> temp_sched = std::make_unique<LinearScheduler>(2.0, 1.0, /* iterations */ 200);
	shn.run( /* Iterations */ 200, temp_sched, uc);
	
	p.block();

	// Create the classifier object to see how the patterns can be classified by representatives
	// after feeding the values. 
	constexpr const auto num_examples_classify = 2000;
	VectorDataset<std::vector<unsigned char>, unsigned int> mnist_sample(num_examples_classify);
	DatasetRepo::load_mnist_vector("vector_mnist_full.data", num_examples_classify, mnist_sample);

	std::cout << "> Constructing the classifier: " << std::endl;
	HopfieldClassifier classifier;
	for (int i = 0; i < 10; ++i)
		classifier.put_mapping(binary_states[i], i);

	// Reject a reconstructed pattern if the agreement is lower than this threshold
	classifier.set_confidence_threshold(0.7);

	dhn.attach_classifier(&classifier);

	// Note: we detached the logger earlier to avoid pointless printing and debug 
	// info for this computation: 

	UpdateConfig group = { UpdatePolicy::GroupUpdate, 15 };

	// To evaluate the performance of our rudimentary classifier with only 10 attractors,
	// construct the confusion matrix. 
	std::vector<double> confusion_matrix(10 * 11);
	constexpr const auto rejected_class_id = 10;

	std::cout << "> Intializing the confusion matrix: " << std::endl;
	for (int i = 0; i < 10; ++i)
		for (int j = 0; j < 11; ++j)
			confusion_matrix[i * 10 + j] = 0.0;

	BinaryState classification_target;
	classification_target.set_stride_y(MNIST_SIZE);

	for (int example = 0; example < num_examples_classify; ++example) {

		// Binarize the image input to allow to be fed to the network. 
		ImageUtils::background_aware_binarize(
			mnist_sample.ref_of(example).data(), MNIST_SIZE, MNIST_SIZE);
		// Load the classification target. 
		StateUtils::load_state_from_byte_array(
			classification_target,
			mnist_sample.ref_of(example).data(),
			MNIST_SIZE * MNIST_SIZE
		);
		auto& v = mnist_sample.ref_of(example);
		// Run 1 synchronous iterations to obtain a good state
		dhn.feed(classification_target);
		dhn.run(5, group);

		auto cls = dhn.classify();

		// Classify the current state of the network using the attached classifier. 
		const auto true_class = mnist_sample.y_of(example);
		const auto predicted_class = (cls) ? std::get<1>(*(cls)) : rejected_class_id;
		
		confusion_matrix[true_class * 10 + predicted_class] += 1;
	}
	p.context().set_title("Confusion matrix").show_heatmap(confusion_matrix.data(), 10, 11);
	p.block();

	// We can learn more about the performance of the network by visualizing the cross talk term
	// which represents the interference between patterns
	HebbianCrossTalkTermVisualizer<float> cttv(p, MNIST_SIZE * MNIST_SIZE);

	// For  example let us visualize the interference between the similar digits 
	// 7 and 4 and 9
	cttv.compute_cross_talk_view(binary_states[7], {&binary_states[4], &binary_states[9]});
	cttv.show(MNIST_SIZE, MNIST_SIZE);

	// The heatmap represents interference in the 1-step error probability sense, meaning that
	// if the cross talk term exceeds 1 the bit is going to be flipped incorrectly. 
	p.block();
	std::cout << "> Done!" << std::endl;
}