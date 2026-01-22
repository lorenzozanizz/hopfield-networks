#include "hopfield_classifier.hpp"


void HopfieldClassifier::put_mapping(const BinaryState& bs, const HopfieldClassifier::Category cat) {
	classification_mapping.emplace_back(bs, cat);
	categories.insert(cat);
}

std::tuple<BinaryState, HopfieldClassifier::Category>* HopfieldClassifier::classify(const BinaryState& bs) {
	// Iterate through our entire mapping computing the agreement for each known pattern,
	// return the pattern with the highest agreement 
	double max_seen_agreement = 0.0;
	std::tuple<BinaryState, HopfieldClassifier::Category>* max_pair = nullptr;

	for (auto& pair : classification_mapping) {
		double agreement = bs.agreement_score(std::get<0>(pair));
		if (agreement > max_seen_agreement) {
			max_seen_agreement = agreement;
			max_pair = &pair;
		}
	}
	if (max_seen_agreement > threshold)
		return max_pair;
	return nullptr;
}
