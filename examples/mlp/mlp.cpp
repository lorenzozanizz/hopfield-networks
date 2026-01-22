#include <vector>
#include <map>

// -------------- INPUT ROUTINES --------------
#include "io/datasets/dataset.hpp"
#include "io/datasets/repository.hpp"

// -------------- RESERVOIR --------------
#include "math/autograd/variables.hpp"
#include "math/utilities.hpp"
#include "reservoir/reservoir_predictor.hpp"

int main() {

	using namespace autograd;

	Plotter plotter;

	// Create an MSE loss function which uses squared l2 norm between
	// the ground truth value and the predicted value, e.g. 
	// MSE(y, y_hat) = squared_norm( y - y_hat ) 
	ScalarFunction loss_function;
	auto& g = loss_function.generator();
	auto y = g.create_vector_variable(100);
	auto y_hat = g.create_vector_variable(100);
	loss_function = g.squared_l2_norm(g.sub(y, y_hat));

	constexpr const auto NUM_SAMPLES = 2000;

	VectorDataset<Eigen::VectorXf, Eigen::VectorXf> sine_series(NUM_SAMPLES);
	DatasetRepo::load_sine_time_series<float>(
		/* amount */ NUM_SAMPLES, sine_series, 100);

	VectorDataset<Eigen::VectorXf, Eigen::VectorXf> verification_sine(NUM_SAMPLES);
	DatasetRepo::load_sine_time_series<float>(
		/* amount */ NUM_SAMPLES, verification_sine, 100, /* offset */ NUM_SAMPLES);

	// Visualize the first requested trained sequence and ground truth that the network
	// will have to learn
	{
		plotter.context().
			set_title("Input sin function").plot_indexable(sine_series.x_of(0), "Input");
		plotter.context().
			set_title("Ground truth following sin function").plot_indexable(sine_series.y_of(0), "Truth");

		plotter.block();
	}

	std::vector<int> units = { 30, 100, 100, 100 };
	auto relu = Activations<float>::relu;
	auto identity = Activations<float>::identity;

	std::vector<ActivationFunction<float>> acts = { relu , relu, identity };

	MultiLayerPerceptron<float> mlp(units, acts);

	NetworkTrainer<float, MultiLayerPerceptron<float>> trainer(mlp);
	trainer.set_loss_function(&loss_function, y, y_hat);
	trainer.do_log_loss(true);
	trainer.do_log_verification_loss(true);

	constexpr const auto BATCH_SIZE = 25;
	trainer.train(
		50,
		sine_series,
		verification_sine, // no verification dataset
		BATCH_SIZE,
		0.005 //alpha
	);

	std::cout << "> Final verification loss: " <<
		std::setprecision(15) << trainer.compute_loss_over_dataset(verification_sine, BATCH_SIZE) << std::endl;

	trainer.plot_loss(plotter);

	plotter.block();

}