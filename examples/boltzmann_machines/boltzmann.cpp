#include <vector>
#include <functional>

// Restricted Boltzmann machines
#include "boltzmann/restricted_boltzmann_machine.hpp"
#include "boltzmann/deep_belief_network.hpp"
#include "boltzmann/boltzmann_logger.hpp"

// Io routines
#include "io/plot/plot.hpp"
#include "io/datasets/dataset.hpp"
#include "io/datasets/repository.hpp"

#include "math/utilities.hpp"

int main() {

	using namespace Eigen;
	using VectorBoltzmann = VectorXf;
	using Machine = RestrictedBoltzmannMachine<float>;

	// Construct the plotter object
	Plotter p;

	constexpr const auto MNIST_SIZE = 28;
	constexpr const auto dataset_size = 300;

	Machine machine_0(MNIST_SIZE * MNIST_SIZE, 529 /* 23 * 23 */);
	Machine machine_1(529, 361 /* 19 * 19 */);
	Machine machine_2(361, 225 /* 15 * 15*/ );

	std::vector<std::reference_wrapper<Machine>> machines = {
		machine_0, machine_1, machine_2
	};
	for (const auto& machine : machines)
		machine.get().initialize_weights(0.008);

	// Declare the deeb belief network stacking together the restricted boltzmann machines
	// then we will train the network in unsupervised fashion and analyzed the kernels. 
	StackedRBM<float> dbf( machines );

	// Load up the MNIST image dataset and binarize it using a threshold binarization
	// policy
	VectorCollection<VectorBoltzmann>  dataset(dataset_size);
	DatasetRepo::load_mnist_eigen<float>(
		"vector_mnist.data", dataset_size, dataset, /* binarization threshold */ 127, 0.0, 1.0);

	std::cout << "> Loaded the dataset: " << dataset.size() << " elements. " << std::endl;

	// Initialize eigen parallelism with the number of hardware threads
	// on the running hardware
	Utilities::eigen_init_parallel(-1);
	std::cout << "> Processing with " << Utilities::eigen_get_num_threads() << " threads on Eigen";
	dbf.unsupervised_train(
		/* Number of iterations */ 40,
		dataset,
		/* Mini batch size */ 25,
		/* Learning rate */ 0.02,
		/* K parameter for CD-K divergence algo */ 10
	);

	std::vector<unsigned int> num_kernels_to_plot = { 9, 9, 9 };

	dbf.visualize_kernels_depthwise(
		p, 3, 3, MNIST_SIZE, MNIST_SIZE, num_kernels_to_plot
	);

	p.block();

}