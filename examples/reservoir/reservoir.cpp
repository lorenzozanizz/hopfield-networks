#include <vector>
#include <memory>

// -------------- RESERVOIR --------------
#include "math/autograd/variables.hpp"
#include "math/utilities.hpp"
#include "reservoir/reservoir.hpp"
#include "reservoir/reservoir_predictor.hpp"
#include "reservoir/reservoir_logger.hpp"

// -------------- INPUT ROUTINES --------------
#include "io/datasets/dataset.hpp"
#include "io/datasets/repository.hpp"
#include "io/image/images.hpp"

// -------------- PLOTTING --------------
#include "io/plot/plot.hpp"
#include "utils/timing.hpp"

int main() {

	using namespace Eigen;
	using Vector = Eigen::VectorXf;

	constexpr const auto input_size = 25;
	constexpr const auto reservoir_size = 625;
	Reservoir<float> reservoir(
		/* Input size */ input_size, /* Reservoir inner state size */ reservoir_size);

	// Initialize the reservoir internal weights and input weights, ensuring that
	// the spectral radius of the reservoir is near 1 to avoid divergence of
	// the state of the reservoir for long sequences. 
	std::cout << "> Initializing the reservoir weights! " << std::endl;
	reservoir.initialize_echo_weights(0.1, SamplingType::Uniform, /* spectral_radius_desired */ 0.5);
	reservoir.initialize_input_weights(SamplingType::Normal);

	// Declare a plotter object. 
	Plotter p;

	// We attach the logger to monitor the activity of the reservoir 
	std::cout << "> Constructing the logger... " << std::endl;
	ReservoirLogger<float> logger; {
		logger.set_collect_norm(true);
		logger.set_collect_states(true, "res_states.gif", /* width render */ 25, /* height render */ 25);
		logger.finally_plot(true);
		logger.assign_plotter(&p);
	}

	reservoir.attach_logger(&logger);

	// We can visualize the activity of the reservoir by plotting its echo
	// states over time and monitoring the norm of the states, to ensure that
	// the dynamics of the reservoir is stable and to visualize the 
	// extreme nonlinearity.

	Vector input(input_size);
	for (int i = 0; i < input_size; ++i)
		input(i) = std::sin(i * 0.3);

	std::cout << "> Hinting off logger... " << std::endl;
	reservoir.begin_run();
	reservoir.feed(input);
	std::cout << "> Starting the echo run " << std::endl;
	for (int i = 0; i < 20; ++i) {
		reservoir.run();
	}
	reservoir.end_run();

	// The logger will implicitly visualize the norm and register the 
	// states into the "res_states.gif" result
	// p.block();

	// Now we present an application of the capability of the reservoir as a
	// nonlinear chaotic representation of sequences.

	/* ---- CLASSIFICATIONS OF THE ONE HOT VECTORS: ----
		 ( 1 0 0 0 0 ) -> Normal beat
		 ( 0 1 0 0 0 ) -> Supraventricular ectopic beat
		 ( 0 0 1 0 0 ) -> Ventricular ectopic beat
		 ( 0 0 0 1 0 ) -> Fusion beat
		 ( 0 0 0 0 1 ) -> Unknown beat (classification above not applicable)

		 https://www.kaggle.com/datasets/josegarciamayen/mit-bih-arrhythmia-dataset-preprocessed
	*/

	constexpr const auto BATCH_SIZE = 1000;
	constexpr const auto NUM_SAMPLES = 15000;

	std::cout << "> Reading the dataset" << std::endl;
	VectorDataset<Vector, Vector> ecg_series(NUM_SAMPLES);
	DatasetRepo::load_mit_bih("mitbih_combined_records.csv", NUM_SAMPLES, ecg_series);

	VectorDataset<Vector, Vector> ecg_embedded_repr(NUM_SAMPLES);

	VectorDataset<Vector, Vector> verification_ecg_series(3000);
	DatasetRepo::load_mit_bih("mitbih_combined_records.csv", 3000, verification_ecg_series, /* offset */ NUM_SAMPLES);

	VectorDataset<Vector, Vector> verification_ecg_embedded_repr(3000);


	// This call to the reservoir maps each vector in the collection ecg_series
	// into a non linear embedding obtained from feeding windows of size 
	// reservoir_input_size into the reservoir dynamics. 

	reservoir.resize(input_size, reservoir_size, BATCH_SIZE);
	reservoir.detach_logger(&logger);
	std::cout << "> Begin mapping the dataset into nonlinear embedding " << std::endl;

	Utilities::eigen_init_parallel(-1);
	Eigen::setNbThreads(5);

	// On my architecture, 5 threads is optimal for this neural network architecture. 
	// It is probably a balance between thread keeping overhead and matmul execution. 
	std::cout << "> Vectorization in use: " << Eigen::SimdInstructionSetsInUse() << std::endl;
	std::cout << "> Eigen reports threads = " << Eigen::nbThreads() << std::endl;

	SegmentTimer timer; {
		auto scoped = timer.scoped("Mapping");
		reservoir.map(ecg_series, ecg_embedded_repr, BATCH_SIZE,
			/* Sequenze size, from keggle */ 187);
		reservoir.map(verification_ecg_series, verification_ecg_embedded_repr, BATCH_SIZE,
			/* Sequenze size, from keggle */ 187);
	}
	std::cout << "> Done!" << std::endl;
	timer.print();

	// Now declare the machinery for the neural network classifier to test the
	// efficacy of the reservoir 
	std::vector<int> units = { reservoir_size, 2000, 1000, 5 };
	auto relu = Activations<float>::relu;
	// We keep an identity activation at the final layer to run the cross entropy soft max
	// loss function for the categorical problem
	auto identity = Activations<float>::identity;
	std::vector<ActivationFunction<float>> acts = { relu , relu, identity };

	MultiLayerPerceptron<float> mlp(units, acts);

	using namespace autograd;

	ScalarFunction loss_function;
	auto y = loss_function.generator().create_vector_variable(5);
	auto y_hat = loss_function.generator().create_vector_variable(5);
	loss_function = loss_function.generator().smce_logits_true(y, y_hat);

	VectorFunction deriv;
	loss_function.derivative(deriv, y);

	NetworkTrainer<float, MultiLayerPerceptron<float>> trainer(mlp);
	trainer.set_loss_function(&loss_function, y, y_hat);
	trainer.do_log_loss(true);
	trainer.do_log_verification_loss(true);

	std::cout << "> Initial verification loss: " << trainer.compute_loss_over_dataset(verification_ecg_embedded_repr, BATCH_SIZE) << std::endl;

	trainer.train(
		80,
		ecg_embedded_repr,
		verification_ecg_embedded_repr, // no verification dataset
		250,
		0.001
	); 

	trainer.plot_loss(p);	
	p.block();

	std::cout << "Final verification loss: " << trainer.compute_loss_over_dataset(verification_ecg_embedded_repr, BATCH_SIZE) << std::endl;

}