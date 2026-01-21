#include "repository.hpp"

namespace DatasetRepo {

	void load_mnist_ten_categories(
		std::string path,
		VectorDataset<std::vector<unsigned char>, unsigned int>& values
	) {
		DatasetRepo::load_mnist_vector(path, 10, values);
	}

	void load_mnist_vector(
		std::string path,
		unsigned int amount,
		VectorDataset<std::vector<unsigned char>, unsigned int>& values
	) {
		// Load all the vector in the input dataset. The standard file format for
				// the vectordataset is a human-readable list of numbers. 
		values.reserve(amount);

		std::ifstream in(path);
		if (!in)
			throw std::runtime_error("Could not open the file stream to load the mnist dataset");

		std::string line;
		int file_amount = 0;
		unsigned int read_amt = 0;
		unsigned int collected_keys = 0;

		// Read "amount: N" 
		if (std::getline(in, line)) {
			std::istringstream iss(line);
			std::string dummy; // dummy = "amount:" 
			iss >> dummy >> file_amount;
		}

		// Variables to temporarily fill while reading the file
		std::vector<unsigned char> vector(28 * 28);
		unsigned int id, label;
		std::string dummy;

		while (std::getline(in, line)) {
			// THIS WAS UPDATED TO FIX A BUG IN WHICH LABELS WOULD GET MESSED UP
			// Skip away empty lines and comments 
			if (line.empty() || line.rfind("#", 0) == 0 || line.rfind("\n", 0) == 0)
				continue;
			else if (line.rfind("label", 0) == 0) {
				// --- Label line parsing 
				std::istringstream iss(line);
				iss >> dummy >> label; // dummy = "label:" 
				collected_keys += 1;
			}
			else if (line.rfind("id", 0) == 0) {
				// --- Id line parsing
				std::istringstream iss(line);
				iss >> dummy >> id; // dummy = "id:"
				collected_keys += 1;
			}
			else if (line.rfind("data", 0) == 0) {
				// --- Data line parsing
				collected_keys += 1;
				std::istringstream iss(line);
				iss >> dummy;
				// "data:" 
				unsigned int value;
				for (int i = 0; i < 28 * 28; ++i) {
					iss >> value;
					vector[i] = value;
				}
			}
			if (collected_keys == 3) {
				// Reset the MNIST image reading context, to allow for variable order
				// of data storage. 
				collected_keys = 0;

				values.add_sample(vector, label, id);
				read_amt++;
				if (read_amt >= amount)
					break;
			}
		}
		in.close();
	}

}
