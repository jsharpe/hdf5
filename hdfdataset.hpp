#ifndef hdfdatasetH
#define hdfdatasetH

#include <hdf5/slab.hpp>
#include <hdf5/hdf5/traits.hpp>
#include <boost/shared_ptr.hpp>

namespace hdf {
  template<class HDFImpl=HDF5Traits>
  class HDFDataSet {
  public:
    HDFDataSet(boost::shared_ptr<typename HDFImpl::dataset_type> dataset)
      : dataset(dataset)
    {}
    
    template<typename Type>
    void
    writeData(const Type &data)
    {
      HDFImpl::write_dataset(*dataset, data);
    }

    template<int order, typename Type>
    void
    writeData(const Type &data, const Slab<order, HDFImpl> & mem)
    {
      HDFImpl::write_dataset(*dataset, data, mem);
    }
    
    template<typename Type>
    void
    readData(std::vector<Type> & data)
    {
      HDFImpl::read_dataset(*dataset, data);
    }

    template<int order, typename Type>
    void
    readData(Type &data, const Slab<order, HDFImpl> & mem)
    {
      HDFImpl::read_dataset(*dataset, data, mem);
    }


    template<typename Type>
    void
    readData(Type &data)
    {
      HDFImpl::read_dataset(*dataset, data);
    }
    
    template<class Type>
    operator Type()
    {
    }
  private:
    boost::shared_ptr<typename HDFImpl::dataset_type> dataset;
  };
}

#endif
