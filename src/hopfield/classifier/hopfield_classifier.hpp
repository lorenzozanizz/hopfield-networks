#pragma once
#ifndef HOPFIELD_CLASSIFIER_HPP

#include <map>
#include <set>
#include <vector>

#include "../states/binary.hpp"

/**
* @brief This class functions as an attachable component to the main hopfield network,
* acts as a selector repositories between a set of known mappings state->categories.
*/
class HopfieldClassifier {
	// Assume for simplicity that the labels are positve integers, as the
	// majority of cases. 
	using Category = int;

	std::vector<std::tuple<BinaryState, Category>> classification_mapping; 
	std::set<Category> categories; 

	// Refuse to classify a value if the threhsold is too low. 
	double threshold;

public:

	HopfieldClassifier() : threshold(0.0) { }

	// When the model refuses to classify, this rejection category is passed instead. 
	static constexpr const int Rejected = -1;

	void put_mapping(const BinaryState& bs, const Category cat);

	void set_confidence_threshold(double confidence) {
		threshold = confidence;
	}
	
	bool can_classify(const Category cat) {
		// Returns true if it's possible for this classifier to classify a pattern with the
		// category passed as input. 
		return categories.count(cat) > 0;
	}

	std::tuple<BinaryState, Category>* classify(const BinaryState& bs);

	std::set<Category>& get_categories() {
		return categories; 
	}

protected:

};

#endif