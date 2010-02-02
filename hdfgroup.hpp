#ifndef hdfgroupH
#define hdfgroupH

#include <hdf5/hdf5/traits.hpp>
#include <hdf5/hdfdataset.hpp>
#include <hdf5/hdfattribute.hpp>
#include <boost/shared_ptr.hpp>

namespace hdf {
  template<typename>
  class HDFFile;
  
  template<class HDFImpl=HDF5Traits>
  class HDFGroup {
    HDFGroup(boost::shared_ptr<typename HDFImpl::group_type> group) : group(group)
    {}
  protected:
    HDFGroup() {};
    void initFileGroup(typename HDFImpl::file_handle_type &file, const boost::filesystem::path & path = boost::filesystem::path("/")) {
      group = HDFImpl::openGroup(file, path);
    }
  public:
    HDFGroup(HDFFile<HDFImpl> & file, const boost::filesystem::path &path) {
      group = HDFImpl::openGroup(file, path);
    }
    
    boost::shared_ptr<HDFDataSet<HDFImpl> >
    openDataset(const boost::filesystem::path & path) {
      return boost::shared_ptr<HDFDataSet<HDFImpl> >
	(new HDFDataSet<HDFImpl>
	 (HDFImpl::openDataSet(*group, path)));
    }
    
    boost::shared_ptr<HDFGroup<HDFImpl> >
    openGroup(const boost::filesystem::path & path) {
      return boost::shared_ptr<HDFGroup<HDFImpl> >
	(new HDFGroup<HDFImpl>
	 (HDFImpl::openGroup(*group, path)));
    }
    
    boost::shared_ptr<HDFGroup<HDFImpl> >
    createGroup(const boost::filesystem::path & path) {
      return boost::shared_ptr<HDFGroup<HDFImpl> >
	(new HDFGroup<HDFImpl>
	 (HDFImpl::createGroup(*group, path)));
    }
    
    template<typename Type>
    boost::shared_ptr<HDFAttribute<HDFImpl> >
    createAttribute(const std::string &name, std::vector<hsize_t> dims, std::vector<hsize_t> maxdims) {
      return boost::shared_ptr<HDFAttribute<HDFImpl> >
	(new HDFAttribute<HDFImpl>
	 (HDFImpl::template createAttribute<Type>(*group, name, dims, maxdims)));
    }
    
    boost::shared_ptr<HDFAttribute<HDFImpl> >
    openAttribute(const std::string &name) {
      return boost::shared_ptr<HDFAttribute<HDFImpl> >
	(new HDFAttribute<HDFImpl>
	 (HDFImpl::openAttribute(*group, name)));
    }
    
    template<typename Type>
    boost::shared_ptr<HDFDataSet<HDFImpl> >
    createDataset(const boost::filesystem::path & path, std::vector<hsize_t> dims, std::vector<hsize_t> maxdims) {
      return boost::shared_ptr<HDFDataSet<HDFImpl> >
	(new HDFDataSet<HDFImpl>
	 (HDFImpl::template createDataSet<Type>(*group, path, dims)));
    }
    
    template<typename Type>
    boost::shared_ptr<HDFDataSet<HDFImpl> >
    writeDataset(const boost::filesystem::path & path, const std::vector<Type> & data) {
      std::vector<hsize_t> dims = HDFImpl::getDims(data);
      boost::shared_ptr<HDFDataSet<HDFImpl> > dataset = createDataset<Type>(path, dims, dims);
      dataset->writeData(data);
      return dataset;
    }
  private:
    boost::shared_ptr<typename HDFImpl::group_type> group;
  };
}

#endif

