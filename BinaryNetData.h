


#pragma once


#include <vector>
#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/json.hpp>


struct LutData {
	std::vector<int>	connect;
	std::vector<bool>	lut;

	template <class Archive>
	void serialize(Archive &archive, std::uint32_t const version)
	{
		archive(cereal::make_nvp("connect", connect));
		archive(cereal::make_nvp("lut", lut));
	}
};


struct BinaryLayerData {
	std::vector<LutData>	node;
	
	template <class Archive>
	void serialize(Archive &archive, std::uint32_t const version)
	{
		archive(cereal::make_nvp("node", node));
	}
};


struct BinaryNetData {
	int								input_num;
	std::vector<BinaryLayerData>	layer;
	
	template <class Archive>
	void serialize(Archive &archive, std::uint32_t const version)
	{
		archive(cereal::make_nvp("input_num", input_num));
		archive(cereal::make_nvp("layer", layer));
	}
};


