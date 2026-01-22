#include <vector>
#include <functional>

// ------- RESTRICTED BOLTZMANN MACHINES -------
#include "boltzmann/restricted_boltzmann_machine.hpp"
#include "boltzmann/deep_belief_network.hpp"
#include "boltzmann/boltzmann_logger.hpp"

// ------- IO ROUTINES -------
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
	constexpr const auto dataset_size = 10000;

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
		"vector_mnist_full.data", dataset_size, dataset, /* binarization threshold */ 127, 0.0, 1.0);

	std::cout << "> Loaded the dataset: " << dataset.size() << " elements. " << std::endl;

	// Initialize eigen parallelism with the number of hardware threads
	// on the running hardware
	Utilities::eigen_init_parallel(-1);

	Eigen::setNbThreads(6);
	std::cout << "> Processing with " << Utilities::eigen_get_num_threads() << " threads on Eigen";
	SegmentTimer timer;
	{
		auto scoped = timer.scoped("Training");
		dbf.unsupervised_train(
			/* Number of iterations */ 40,
			dataset,
			/* Mini batch size */ 250,
			/* Learning rate */ 0.02,
			/* K parameter for CD-K divergence algo */ 13
		);
	}
	timer.print();

	std::vector<unsigned int> num_kernels_to_plot = { 9, 9, 9};

	dbf.visualize_kernels_depthwise(
		p, 3, 3, MNIST_SIZE, MNIST_SIZE, num_kernels_to_plot
	);
	// Visualize how the kernels refine semantically the more in depth in the 
	// stack of RBMS we go. 
	p.wait();

	// We can observe the tendency of the RBMS to reconstruct and sample from
	// the initial distribution by feeding a random pattern in the
	// RBM initially and observing how the CD-K algorithm brings a new pattern
	// into existance. 

	BoltzmannLogger<float> logger;
	logger.set_collect_states(true, "boltzmann_states.gif",
		MNIST_SIZE, MNIST_SIZE);

	machine_0.attach_logger(&logger);
	machine_0.random_visible(/* sparsity */ 0.15);
	machine_0.run_cd_k(5, false, 0.8);

}