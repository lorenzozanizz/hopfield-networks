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

// Restricted Boltzmann machines

#include "io/plot/plot.hpp"
#include "io/gif/gif.hpp"
#include "io/image/images.hpp"

#include "math/matrix/matrix_ops.hpp"

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
		g.squared_norm( (g.sub(u, 1.0) )),
		g.multiply(lambda, g.squared_norm(u) )
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


	StateUtils::load_state_from_image(bs1, img1_r, /* binarize */ true);

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
	logger.set_recording_stream(save_file, /* close_after */ true);

	dhn.attach_logger(&logger);
	logger.set_collect_states(true, "states.gif");
	logger.set_collect_energy(true);
	logger.set_collect_temperature(true);
	logger.set_collect_order_parameter(true);

	logger.finally_write_last_state_png(true, "last_state.png");
	logger.finally_plot_data(true);
	
	StateUtils::perturb_state(bs1, /* Noise intensity 0 to 1*/ 0.15);
	std::cout << "Plotting state!" << std::endl;
	StateUtils::plot_state(p, bs1);

	UpdateConfig uc = {
		UpdatePolicy::Asynchronous
	};


	// Instruct the network that we wish to interpret the state as a 40x40 raster.
	dhn.set_state_strides(40);
	dhn.set_reference_state(bs1);
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

// Just create the folder...
int main() {
	
	autograd_compile();
	io_utils_compile();
	// hopfield_compile();
}
	