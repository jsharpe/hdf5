#ifndef hdfdatasetH
#define hdfdatasetH

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
    
    template<typename Type>
    void
    readData(std::vector<Type> & data)
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
