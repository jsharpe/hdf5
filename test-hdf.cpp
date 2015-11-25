#include <hdf5/hdffile.hpp>
#include "gtest/gtest.h"

class Hdf5Test: public ::testing::Test {
protected:
	const std::string file_name_ { "test.h5" };
};

TEST_F(Hdf5Test, CreateFile) {
	hdf::HDFFile<> file(file_name_, hdf::HDFFile<>::truncate);
}

TEST_F(Hdf5Test, CreateDataset) {
	hdf::HDFFile<> file(file_name_, hdf::HDFFile<>::truncate);
	std::vector<hsize_t> dims;
	dims.resize(2);
	dims[0] = 20;
	dims[1] = 5;
	hdf::Slab<2> filespace(dims);
	auto datasetint = file.createDataset<int>(
			"/test", filespace);
	auto datasetfloat = file.createDataset<
			float>("/test-float", filespace);
	auto datasetdouble = file.createDataset<double>("/test-double", filespace);
}

TEST_F(Hdf5Test, OpenDataset) {
	hdf::HDFFile<> file(file_name_, hdf::HDFFile<>::truncate);
	{
		std::vector<hsize_t> dims;
		dims.resize(2);
		dims[0] = 15;
		dims[1] = 5;
		hdf::Slab<2> filespace(dims);
		auto datasetint = file.createDataset<int>("/test", filespace);
	}
	auto dataset = file.openDataset("/test");
}

TEST_F(Hdf5Test, CreateGroup) {
	hdf::HDFFile<> file(file_name_, hdf::HDFFile<>::truncate);
	auto group = file.createGroup("/test-group");
}

TEST_F(Hdf5Test, OpenGroup) {
	hdf::HDFFile<> file(file_name_, hdf::HDFFile<>::truncate);
	{
		auto group = file.createGroup("/test-group");
	}
	auto group = file.openGroup("/test-group");
}

TEST_F(Hdf5Test, CreateAttribute) {
	hdf::HDFFile<> file(file_name_, hdf::HDFFile<>::truncate);
	auto group = file.createGroup("/test-group");
	std::vector<hsize_t> dims;
	dims.resize(2);
	dims[0] = 1;
	dims[1] = 3;

	group->createAttribute<int>("attr", dims);
}

TEST_F(Hdf5Test, OpenAttribute) {
	hdf::HDFFile<> file(file_name_, hdf::HDFFile<>::truncate);
	auto group = file.createGroup("/test-group");
	std::vector<hsize_t> dims;
	dims.resize(2);
	dims[0] = 1;
	dims[1] = 3;
	group->createAttribute<int>("attr", dims);
	group->openAttribute(std::string("attr"));
}

TEST_F(Hdf5Test, WriteAttribute) {
	hdf::HDFFile<> file(file_name_, hdf::HDFFile<>::truncate);
	auto group = file.createGroup("/test-group");
	std::vector<hsize_t> dims(1, 1);

	group->createAttribute<int>("attr", dims);
	auto attr = group->openAttribute(std::string("attr"));
	attr->writeData(10);
	int val;
	attr->readData(val);
	assert(val == 10);
}

TEST_F(Hdf5Test, WriteDataset) {
	{
		hdf::HDFFile<> file(file_name_, hdf::HDFFile<>::truncate);

		std::vector<hsize_t> dims;
		dims.resize(2);
		dims[0] = 3;
		dims[1] = 2;
		hdf::Slab<2> filespace(dims);
		auto datasetint = file.createDataset<
				int>("/test", filespace);
	}
	hdf::HDFFile<> file(file_name_);

	auto dataset = file.openDataset("/test");

	std::vector<int> testdata(6);
	testdata[0] = 1;
	testdata[1] = 2;
	testdata[2] = 3;
	testdata[3] = 4;
	testdata[4] = 5;
	testdata[5] = 6;
	dataset->writeData(testdata);

	{
		std::vector<hsize_t> dims;
		dims.resize(2);
		dims[0] = 2;
		dims[1] = 3;
		hdf::Slab<2> filespace(dims);
		auto datasetint = file.createDataset<
				int>("/test2", filespace);
		datasetint->writeData(testdata);

		std::vector<int> readData2;
		datasetint->readData(readData2);
		ASSERT_TRUE(readData2.size() == 6);
		ASSERT_TRUE(readData2[0] == 1);
		ASSERT_TRUE(readData2[1] == 2);
		ASSERT_TRUE(readData2[2] == 3);
		ASSERT_TRUE(readData2[3] == 4);
	}

	std::vector<int> readData;
	dataset->readData(readData);
	ASSERT_TRUE(readData.size() == 6);
	ASSERT_TRUE(readData[0] == 1);
	ASSERT_TRUE(readData[1] == 2);
	ASSERT_TRUE(readData[2] == 3);
	ASSERT_TRUE(readData[3] == 4);
}

#include <boost/fusion/adapted/struct/adapt_struct.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/adapted/std_pair.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/fusion/adapted/std_tuple.hpp>
#include <boost/fusion/include/std_tuple.hpp>

struct test_point {
	test_point() = default;
	test_point(float x, float y, float z) :
			x(x), y(y), z(z) {
	}
	float x, y, z;

	bool operator ==(const test_point & t) const {
		return x == t.x && y == t.y && z == t.z;
	}
};

BOOST_FUSION_ADAPT_STRUCT(test_point, (float, x) (float, y) (float, z));

TEST_F(Hdf5Test, WriteHomoPair) {
	{
		hdf::HDFFile<> file(file_name_, hdf::HDFFile<>::truncate);
		std::vector<std::pair<float, float> > pairs = { {1.0f, 2.5f}, {1.2f, 3.5f} };
		file.writeDataset("/test1", pairs);
	}
	hdf::HDFFile<> file(file_name_);
	auto datasetpairs = file.openDataset("/test1");

	std::vector<std::pair<float, float> > pairs;
	datasetpairs->readData(pairs);
	ASSERT_EQ(2, pairs.size());
	ASSERT_EQ(std::make_pair(1.0f, 2.5f), pairs[0]);
	ASSERT_EQ(std::make_pair(1.2f, 3.5f), pairs[1]);

}


TEST_F(Hdf5Test, WriteHeteroPair) {
	{
		hdf::HDFFile<> file(file_name_, hdf::HDFFile<>::truncate);
		std::vector<std::pair<int, float> > pairs = { {1, 2.5f}, {1, 3.5f}, {2, 1.4f}, {5, 2.6f} };
		file.writeDataset("/test1", pairs);
	}
	hdf::HDFFile<> file(file_name_);
	auto datasetpairs = file.openDataset("/test1");

	std::vector<std::pair<int, float> > pairs;
	datasetpairs->readData(pairs);
	ASSERT_EQ(4, pairs.size());
	ASSERT_EQ(std::make_pair(1, 2.5f), pairs[0]);
	ASSERT_EQ(std::make_pair(1, 3.5f), pairs[1]);

}
TEST_F(Hdf5Test, WriteHomoTuple) {
	{
		hdf::HDFFile<> file(file_name_, hdf::HDFFile<>::truncate);
		std::vector<std::tuple<int, int, int> > tuples = { {1, 2, 3}, {4, 5, 6} };
		file.writeDataset("/test1", tuples);
	}
	hdf::HDFFile<> file(file_name_);
	auto datasetpairs = file.openDataset("/test1");

	std::vector<std::tuple<int, int, int> > tuples;
	datasetpairs->readData(tuples);
	ASSERT_EQ(2, tuples.size());
	ASSERT_EQ(std::make_tuple(1, 2, 3), tuples[0]);
	ASSERT_EQ(std::make_tuple(4, 5, 6), tuples[1]);
}


TEST_F(Hdf5Test, WriteHeteroTuple) {
	{
		hdf::HDFFile<> file(file_name_, hdf::HDFFile<>::truncate);
		std::vector<std::tuple<int, float, char> > tuples = { {1, 2.5f, 'a'}, {1, 3.5f, 'b'} };
		file.writeDataset("/test1", tuples);
	}
	hdf::HDFFile<> file(file_name_);
	auto datasetpairs = file.openDataset("/test1");

	std::vector<std::tuple<int, float, char> > tuples;
	datasetpairs->readData(tuples);
	ASSERT_EQ(2, tuples.size());
	ASSERT_EQ(std::make_tuple(1, 2.5f, 'a'), tuples[0]);
	ASSERT_EQ(std::make_tuple(1, 3.5f, 'b'), tuples[1]);
}


TEST_F(Hdf5Test, WriteStructure) {
	{
		hdf::HDFFile<> file(file_name_, hdf::HDFFile<>::truncate);

		std::vector<test_point> points(4);
		points[0] = test_point(0, 0, 1);
		points[1] = test_point(1, 0, 0);
		points[2] = test_point(1, 0, 1);
		points[3] = test_point(0, 1, 0);
		auto datasetpoints = file.writeDataset("/test", points);
	}

	hdf::HDFFile<> file(file_name_);
	auto datasetpoints = file.openDataset(
			"/test");
	std::vector<test_point> read;
	datasetpoints->readData(read);
	ASSERT_EQ(4, read.size());
	ASSERT_EQ(test_point(0, 0, 1), read[0]);
	ASSERT_EQ(test_point(1, 0, 0), read[1]);
	ASSERT_EQ(test_point(1, 0, 1), read[2]);
	ASSERT_EQ(test_point(0, 1, 0), read[3]);
}
