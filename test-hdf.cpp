#include <hdf5/hdffile.hpp>


void test_file_creation() {
    hdf::HDFFile<> file("test.h5", hdf::HDFFile<>::truncate);
}

void test_dataset_creation() {
    hdf::HDFFile<> file("test.h5", hdf::HDFFile<>::truncate);
    std::vector<hsize_t> dims;
    dims.resize(2);
    dims[0] = 20;
    dims[1] = 5;
    hdf::Slab<2> filespace(dims);
    auto datasetint = file.createDataset<int>("/test", filespace);
    auto datasetfloat = file.createDataset<float>("/test-float", filespace);
    auto datasetdouble = file.createDataset<double>("/test-double", filespace);
}

void test_dataset_open() {
    hdf::HDFFile<> file("test.h5", hdf::HDFFile<>::truncate);
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

void test_group_create() {
    hdf::HDFFile<> file("test.h5", hdf::HDFFile<>::truncate);
    auto group = file.createGroup("/test-group");
}

void test_group_open() {
    hdf::HDFFile<> file("test.h5", hdf::HDFFile<>::truncate);
    {
        auto group = file.createGroup("/test-group");
    }
    auto group = file.openGroup("/test-group");
}

void test_attribute_create() {
    hdf::HDFFile<> file("test.h5", hdf::HDFFile<>::truncate);
    auto group = file.createGroup("/test-group");
    std::vector<hsize_t> dims;
    dims.resize(2);
    dims[0] = 1;
    dims[1] = 3;

    group->createAttribute<int>("attr", dims);
}

void test_attribute_open() {
    hdf::HDFFile<> file("test.h5", hdf::HDFFile<>::truncate);
    auto group = file.createGroup("/test-group");
    std::vector<hsize_t> dims;
    dims.resize(2);
    dims[0] = 1;
    dims[1] = 3;

    group->createAttribute<int>("attr", dims);

    group->openAttribute(std::string("attr"));
}

void test_attribute_write() {
    hdf::HDFFile<> file("test.h5", hdf::HDFFile<>::truncate);
    auto group = file.createGroup("/test-group");
    std::vector<hsize_t> dims(1,1);

    group->createAttribute<int>("attr", dims);
    auto attr = group->openAttribute(std::string("attr"));
    attr->writeData(10);
    int val;
    attr->readData(val);
    assert(val == 10);
}

void test_dataset_write() {
    {
        hdf::HDFFile<> file("test.h5", hdf::HDFFile<>::truncate);

        std::vector<hsize_t> dims;
        dims.resize(2);
        dims[0] = 3;
        dims[1] = 2;
        hdf::Slab<2> filespace(dims);
        auto datasetint = file.createDataset<int>("/test", filespace);
    }
    hdf::HDFFile<> file("test.h5");

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
        auto datasetint = file.createDataset<int>("/test2",filespace);
        datasetint->writeData(testdata);

        std::vector<int> readData2;
        datasetint->readData(readData2);
        assert(readData2.size() == 6);
        assert(readData2[0] == 1);
        assert(readData2[1] == 2);
        assert(readData2[2] == 3);
        assert(readData2[3] == 4);
    }

    std::vector<int> readData;
    dataset->readData(readData);
    assert(readData.size() == 6);
    assert(readData[0] == 1);
    assert(readData[1] == 2);
    assert(readData[2] == 3);
    assert(readData[3] == 4);
}

#include <boost/fusion/adapted/struct/adapt_struct.hpp>
#include <boost/fusion/include/adapt_struct.hpp>

struct test_point {
    test_point() {}
    test_point(float x, float y, float z) : x(x), y(y), z(z) {}
    float x, y, z;

    bool operator ==(const test_point & t) const {
        return x == t.x && y == t.y && z == t.z;
    }
};

BOOST_FUSION_ADAPT_STRUCT(test_point,
                          (float, x)
                          (float, y)
                          (float, z)
                         );

#include <boost/fusion/adapted/std_pair.hpp>
#include <boost/fusion/include/std_pair.hpp>

void
test_dataset_structure_write() {
    {
        hdf::HDFFile<> file("teststruct.h5", hdf::HDFFile<>::truncate);
        std::vector<std::pair<int, float> > pairs(2);
        pairs[0] = std::make_pair(1, 2.5f);
        pairs[1] = std::make_pair(1, 3.5f);
        file.writeDataset("/test1", pairs);

        std::vector<test_point> points(4);
        points[0] = test_point(0,0,1);
        points[1] = test_point(1,0,0);
        points[2] = test_point(1,0,1);
        points[3] = test_point(0,1,0);
        auto datasetpoints = file.writeDataset("/test", points);
    }

    hdf::HDFFile<> file("teststruct.h5");
    auto datasetpoints = file.openDataset("/test");
    std::vector<test_point> read;
    datasetpoints->readData(read);
    assert(read.size() == 12);
    assert(read[0] == test_point(0,0,1));
    assert(read[1] == test_point(1,0,0));

    auto datasetpairs = file.openDataset("/test1");

    std::vector<std::pair<int, float> > pairs;
    datasetpairs->readData(pairs);
    assert(pairs.size() == 2);
    assert(pairs[0] == std::make_pair(1, 2.5f));
}

int main() {
    test_file_creation();
    test_dataset_creation();
    test_dataset_open();
    test_group_create();
    test_group_open();
    test_attribute_create();
    test_attribute_open();
    test_attribute_write();
    test_dataset_write();
    test_dataset_structure_write();
    return 0;
};
