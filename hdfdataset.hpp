#ifndef hdfdatasetH
#define hdfdatasetH

#include <hdf5/slab.hpp>
#include <hdf5/hdf5/traits.hpp>
#include <stdexcept>

namespace hdf {
template<class HDFImpl=HDF5Traits>
class HDFDataSet : std::enable_shared_from_this<HDFDataSet<HDFImpl> > {
  public:
    HDFDataSet(std::shared_ptr<typename HDFImpl::dataset_type> dataset)
        : dataset(dataset) {
    }

    template<typename Type>
    std::shared_ptr<HDFDataSet<HDFImpl> >
    selectSubset(const std::vector<Type> & mapping) {
        if(mapping.empty()) throw std::runtime_error("No mapping available");
        dataset = dataset->selectSubset(mapping);
        return this->shared_from_this();
    }

    template<typename Type>
    void
    writeData(const Type &data) {
        HDFImpl::write_dataset(*dataset, data);
    }
#ifdef H5_HAVE_PARALLEL
    template<typename Type>
    void
    writeParallelData(const Type &data) {
        HDFImpl::write_parallel_dataset(*dataset, data);
    }
#endif
    template<int order, typename Type>
    void
    writeData(const Type &data, const Slab<order, HDFImpl> & mem) {
        HDFImpl::write_dataset(*dataset, data, mem);
    }

    template<int order, typename Type>
    void
    writeData(const Type &data,
              const Slab<order, HDFImpl> & mem,
              const Slab<order, HDFImpl> & filespace) {
        HDFImpl::write_dataset(*dataset, data, mem, filespace);
    }
#ifdef H5_HAVE_PARALLEL
    template<int order, typename Type>
    void
    writeParallelData(const Type &data, const Slab<order, HDFImpl> & mem) {
        HDFImpl::write_parallel_dataset(*dataset, data, mem);
    }

    template<int order, typename Type>
    void
    writeParallelData(const Type &data,
                      const Slab<order, HDFImpl> & memSpace,
                      const Slab<order, HDFImpl> & fileSpace) {
        HDFImpl::write_parallel_dataset(*dataset, data, memSpace, fileSpace);
    }
#endif
    template<int order, typename Type>
    void
    writeData(const std::vector<Type> &data, const Slab<order, HDFImpl> & mem) {
        HDFImpl::write_dataset(*dataset, &data[0], mem);
    }

    template<typename Type>
    void
    readData(std::vector<Type> & data) {
        HDFImpl::read_dataset(*dataset, data);
    }

    template<int order, typename Type>
    void
    readData(Type *data, const Slab<order, HDFImpl> & mem) {
        HDFImpl::read_dataset(*dataset, data, mem);
    }
    template<int order, typename Type>
    void
    readData(Type *data, const Slab<order, HDFImpl> & mem,
             const Slab<order, HDFImpl> & filespace) {
        HDFImpl::read_dataset(*dataset, data, mem,filespace);
    }
#ifdef H5_HAVE_PARALLEL
    template<int order, typename Type>
    void
    readParallelData(Type *data, const Slab<order, HDFImpl> & mem) {
        HDFImpl::read_parallel_dataset(*dataset, data, mem);
    }

    template<int order, typename Type>
    void
    readParallelData(const Type &data,
                     const Slab<order, HDFImpl> & memSpace,
                     const Slab<order, HDFImpl> & fileSpace) {
        HDFImpl::read_parallel_dataset(*dataset, data, memSpace, fileSpace);
    }
#endif
    std::vector<hsize_t>
    getDimensions() const {
        return dataset->getDataSpace()->getDimensions();
    }

    template<typename Type>
    void
    readData(Type *data) {
        HDFImpl::read_dataset(*dataset, data);
    }

    template<class Type>
    operator Type() {
    }
  private:
    std::shared_ptr<typename HDFImpl::dataset_type> dataset;
};
}

#endif
