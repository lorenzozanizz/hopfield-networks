#pragma once
#ifndef IO_DATASET_DATA_COLLECTION_HPP
#define IO_DATASET_DATA_COLLECTION_HPP

#include <map>
#include <vector>
#include <iostream>

template <typename DataType = float>
class VectorCollection {


};

template <typename DataType = float>
class NamedVectorCollection {

	std::map<std::string, std::vector<DataType>> mapped_data;

	using DataVector = std::vector<DataType>;
	using DataMap = std::map < std::string, std::vector<DataType>>;

public:

	void register_name(const std::string& name, unsigned int reserve = 4) {
		mapped_data[name] = DataVector();
		mapped_data[name].reserve(reserve);
		return;
	}

	inline void append_for(const std::string name, DataType val) {
		mapped_data[name].push_back(val);
	}

	std::vector<DataType&> data_for(const std::string& name) {
		return mapped_data[name];
	}

	unsigned int num_names() const {
		return mapped_data.size();
	}

	unsigned long byte_size() const {
		// Approximate estimate of the data size
		unsigned long bytes = 0;
		for (const auto& v : mapped_data)
			bytes += (v.second.size()) * sizeof(DataType);
		return bytes;
	}

	void clear() {
		for (auto& v : mapped_data)
			v.second.clear();
		mapped_data.clear();
	}
	
	auto begin() const {
		return mapped_data.begin();
	}

	auto end() const {
		return mapped_data.end();
	}

};

namespace DataUtils {

	template <typename DataType>
	void dump_named_data_to_file(const std::string& file_name, const NamedVectorCollection<DataType>& ndc) {
		std::ofstream file_stream(file_name);
		dump_named_data_to_file(file_stream, ndc);
		file_stream.close();
	}

	template <typename DataType>
	void dump_named_data_to_file(std::ofstream& out, const NamedVectorCollection<DataType>& ndc) {
		// Prepare the header section
		out << ndc.num_names() << "\n";
		// format of the type:
		// [num_names]
		// name_1 : size_1...
		// ... data on a single line
		for (const auto& pair : ndc) {
			auto& data_vec = pair.second;
			out << pair.first << " " << data_vec.size() << "\n";
			for (unsigned int i = 0; i < data_vec.size(); ++i)
				out << data_vec[i];
			out << "\n";
		}
		return;
	}


	template <typename DataType>
	void read_named_data_from_file(std::ifstream& in, NamedVectorCollection<DataType>& ndc, bool clear_before = true) {
		// Read the header data section to get all the data
		if (clear_before)
			ndc.clear();
		unsigned int names_amt, name_size;
		std::string name;

		in >> names_amt;
		for (int name_index = 0; name_index < names_amt; ++name_index) {
			in >> name;
			in >> name_size;
			ndc.register_name(name, /* reserve capacity */ name_size);
			auto& data_vec = ndc.data_for(name_size);
			for (int i = 0; i < name_size; ++i) {
				in >> data_vec[i];
			}
		}
		return;
	}	

	template <typename DataType>
	void read_named_data_from_file(std::string& file_name, NamedVectorCollection<DataType>& ndc, bool clear_before = true) {
		std::ifstream file_stream(file_name);
		read_named_data_from_file(file_stream, ndc);
		file_stream.close();
	}

	template <typename DataType>
	void read_data_collection_from_file(const std::string& file_name, VectorCollection< DataType>& ndc) {

	}

}

#endif