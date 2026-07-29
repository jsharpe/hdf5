#ifndef hdfdatasetH
#define hdfdatasetH

#include "slab.hpp"
#include "hdf5/traits.hpp"
#include "hdfattribute.hpp"
#include <stdexcept>

namespace hdf {
template <class HDFImpl = HDF5Traits>
class HDFDataSet {
public:
    HDFDataSet(std::shared_ptr<typename HDFImpl::dataset_type> dataset) : dataset(dataset) {}

    /**
     * Create an attribute attached to this dataset
     */
    template <typename Type>
    std::unique_ptr<HDFAttribute<HDFImpl> > createAttribute(const std::string& name, std::vector<hsize_t> dims) {
        return std::unique_ptr<HDFAttribute<HDFImpl> >(
            new HDFAttribute<HDFImpl>(HDFImpl::template createAttribute<Type>(*dataset, name, dims, dims)));
    }

    /**
     * Open an existing attribute of this dataset
     */
    std::unique_ptr<HDFAttribute<HDFImpl> > openAttribute(const std::string& name) {
        return std::unique_ptr<HDFAttribute<HDFImpl> >(new HDFAttribute<HDFImpl>(HDFImpl::openAttribute(*dataset, name)));
    }

    template <typename Type>
    void writeAttribute(const std::string& name, const Type& data) {
        std::vector<hsize_t> dims(1, 1);
        std::unique_ptr<HDFAttribute<HDFImpl> > attr;
        try {
            attr = createAttribute<Type>(name, dims);
        } catch(AttributeExists&) {
            attr = openAttribute(name);
        }
        attr->writeData(data);
    }

    template <typename Type>
    void readAttribute(const std::string& name, Type& data) {
        openAttribute(name)->readData(data);
    }

    template <typename Type>
    HDFDataSet<HDFImpl>* selectSubset(const std::vector<Type>& mapping) {
        if(mapping.empty())
            throw std::runtime_error("No mapping available");
        dataset = dataset->selectSubset(mapping);
        return this;
    }

    template <typename Type>
    void writeData(const Type& data) {
        HDFImpl::write_dataset(*dataset, data);
    }

#ifdef H5_HAVE_PARALLEL
    template <typename Type>
    void writeParallelData(const Type& data) {
        HDFImpl::write_parallel_dataset(*dataset, data);
    }
#endif

    template <int order, typename Type>
    void writeData(const Type& data, const Slab<order, HDFImpl>& mem) {
        HDFImpl::write_dataset(*dataset, data, mem);
    }

    template <int order, typename Type>
    void writeData(const Type& data, const Slab<order, HDFImpl>& mem, const Slab<order, HDFImpl>& filespace) {
        HDFImpl::write_dataset(*dataset, data, mem, filespace);
    }

#ifdef H5_HAVE_PARALLEL
    template <int order, typename Type>
    void writeParallelData(const Type* data, const Slab<order, HDFImpl>& mem) {
        HDFImpl::write_parallel_dataset(*dataset, data, mem);
    }

    template <int order, typename Type>
    void writeParallelData(const std::vector<Type>& data, const Slab<order, HDFImpl>& mem) {
        HDFImpl::write_parallel_dataset(*dataset, data.data(), mem);
    }

    template <int order, typename Type>
    void writeParallelData(const Type* data, const Slab<order, HDFImpl>& memSpace,
                           const Slab<order, HDFImpl>& fileSpace) {
        HDFImpl::write_parallel_dataset(*dataset, data, memSpace, fileSpace);
    }

    template <int order, typename Type>
    void writeParallelData(const std::vector<Type>& data, const Slab<order, HDFImpl>& memSpace,
                           const Slab<order, HDFImpl>& fileSpace) {
        HDFImpl::write_parallel_dataset(*dataset, data.data(), memSpace, fileSpace);
    }
#endif

    template <int order, typename Type>
    void writeData(const std::vector<Type>& data, const Slab<order, HDFImpl>& mem) {
        HDFImpl::write_dataset(*dataset, data.data(), mem);
    }

    template <int order, typename Type>
    void writeData(const std::vector<Type>& data, const Slab<order, HDFImpl>& mem,
                   const Slab<order, HDFImpl>& filespace) {
        HDFImpl::write_dataset(*dataset, data.data(), mem, filespace);
    }

    template <typename Type>
    void readData(std::vector<Type>& data) {
        HDFImpl::read_dataset(*dataset, data);
    }

    template <int order, typename Type>
    void readData(Type* data, const Slab<order, HDFImpl>& mem) {
        HDFImpl::read_dataset(*dataset, data, mem);
    }

    template <int order, typename Type>
    void readData(Type* data, const Slab<order, HDFImpl>& mem, const Slab<order, HDFImpl>& filespace) {
        HDFImpl::read_dataset(*dataset, data, mem, filespace);
    }

#ifdef H5_HAVE_PARALLEL
    template <int order, typename Type>
    void readParallelData(Type* data, const Slab<order, HDFImpl>& mem) {
        HDFImpl::read_parallel_dataset(*dataset, data, mem);
    }

    template <int order, typename Type>
    void readParallelData(const Type& data, const Slab<order, HDFImpl>& memSpace,
                          const Slab<order, HDFImpl>& fileSpace) {
        HDFImpl::read_parallel_dataset(*dataset, data, memSpace, fileSpace);
    }
#endif

    std::vector<hsize_t> getDimensions() const {
        return dataset->getDataSpace()->getDimensions();
    }

    template <typename Type>
    void readData(Type* data) {
        HDFImpl::read_dataset(*dataset, data);
    }

    hid_t hid() const {
        return dataset->hid();
    }
private:
    std::shared_ptr<typename HDFImpl::dataset_type> dataset;
};
}  // namespace hdf

#endif
