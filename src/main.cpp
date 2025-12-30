#include <cstdio>
#include <memory>
#include <iostream>

#include "hopfield/states/binary.hpp"
#include "hopfield/deterministic/dense_hopfield_network.hpp"
#include "hopfield/deterministic/cross_talk_visualizer.hpp"
#include "hopfield/logger/logger.hpp"
#include "math/autograd/variables.hpp"

#include "io/plot/plot.hpp"
#include "io/gif/gif.hpp"
#include "io/image/images.hpp"

enum NeighbouringStrategy {
	OneDNeighbouring,
	TwoDNeighbouring,
	ThreeDNeighbouring
};


void
autograd_compile() {

	using namespace autograd;
	
	Function func(1); // a scalar function
	auto& g = func.generator();
	auto u = g.create_vector_variable(140);
	const auto expr = g.exponential(g.sum(u, 1.0));

}

void
hopfield_compile() {

	Plotter p;

	DenseHopfieldNetwork<HebbianPolicy> dhn(20 * 20);

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

	std::cout << "Loaded image!";
	dhn.store(bs1);

	StateUtils::load_state_from_image(bs2, img2, /* binarize */ true);
	dhn.store(bs2);

	StateUtils::load_state_from_image(bs3, img3, /* binarize */ true);
	dhn.store(bs3);

	// Create and open a text file 
	auto logger = HopfieldLogger();
	std::ofstream save_file("save_data.txt");
	logger.set_recording_stream(save_file, /* close_after */ true);

	dhn.attach_logger(&logger);
	logger.collect_states(true, "states.gif");
	logger.collect_energy(true);
	logger.collect_temperature(true);
	logger.collect_order_parameter(true);

	logger.finally_write_last_state_png("last_state.png");
	logger.finally_plot_data(true);
	
	StateUtils::perturb_state(bs1, /* Noise intensity 0 to 1*/ 0.15);

	StateUtils::plot_state(p, bs1);

	UpdateConfig uc = {
		UpdatePolicy::OnlineUpdate
	};

	dhn.run(bs1, 20, uc);
	dhn.detach_logger(&logger);

	CrossTalkTermVisualizer cttv(bs3, { &bs2, &bs1 });
	cttv.show();

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
	hopfield_compile();
}
	