#ifndef hdfgroupH
#define hdfgroupH

#include <hdf5/hdf5/traits.hpp>
#include <hdf5/hdfdataset.hpp>
#include <hdf5/hdfattribute.hpp>
#include <hdf5/slab.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem/path.hpp>

namespace hdf {
  template<typename>
  class HDFFile;

  /**
   * Represents a group object in a hdf file
   */
  template<class HDFImpl=HDF5Traits>
  class HDFGroup {
    HDFGroup(boost::shared_ptr<typename HDFImpl::group_type> group) : group(group)
    {}
  protected:
    HDFGroup() {};

    /**
     * Used by HDFFile to act like as a group - it is by default the group "/"
     * but a path can be specified to support mounting files as a group
     * within another file
     */
    void initFileGroup(typename HDFImpl::file_handle_type &file, const boost::filesystem::path & path = boost::filesystem::path("/")) {
      group = HDFImpl::openGroup(file, path, true); //pass true to create here for the case where we are mounting a file in a file where the mount point doesn't yet exist
    }

  public:
    /**
     * Open a group from an existing file
     */
    HDFGroup(HDFFile<HDFImpl> & file, const boost::filesystem::path &path) {
      group = HDFImpl::openGroup(file, path);
    }
        
    /**
     * Open an existing group
     * Create the group if it doesn't exist and create=true
     */
    boost::shared_ptr<HDFGroup<HDFImpl> >
    openGroup(const boost::filesystem::path & path, bool create=false) {
      return boost::shared_ptr<HDFGroup<HDFImpl> >
	(new HDFGroup<HDFImpl>
	 (HDFImpl::openGroup(*group, path, create)));
    }
    
    /**
     * Create a new group
     */
    boost::shared_ptr<HDFGroup<HDFImpl> >
    createGroup(const boost::filesystem::path & path) {
      return boost::shared_ptr<HDFGroup<HDFImpl> >
	(new HDFGroup<HDFImpl>
	 (HDFImpl::createGroup(*group, path)));
    }
    
    /**
     * Create an attribute attached to this group
     */
    template<typename Type>
    boost::shared_ptr<HDFAttribute<HDFImpl> >
    createAttribute(const std::string &name, std::vector<hsize_t> dims) {
      return boost::shared_ptr<HDFAttribute<HDFImpl> >
	(new HDFAttribute<HDFImpl>
	 (HDFImpl::template createAttribute<Type>(*group, name, dims, dims)));
    }
    
    /**
     * Open an existing attribute of this group
     */
    boost::shared_ptr<HDFAttribute<HDFImpl> >
    openAttribute(const std::string &name) {
      return boost::shared_ptr<HDFAttribute<HDFImpl> >
	(new HDFAttribute<HDFImpl>
	 (HDFImpl::openAttribute(*group, name)));
    }
    
    /**
     * Open an existing dataset
     */
    boost::shared_ptr<HDFDataSet<HDFImpl> >
    openDataset(const boost::filesystem::path & path) {
      return boost::shared_ptr<HDFDataSet<HDFImpl> >
	(new HDFDataSet<HDFImpl>
	 (HDFImpl::openDataSet(*group, path)));
    }

    /**
     * Create a new dataset as a child of this group
     * dims specifies the dimensions of the dataset in the file
     */
    template<typename Type, int order>
    boost::shared_ptr<HDFDataSet<HDFImpl> >
    createDataset(const boost::filesystem::path & path, const Slab<order, HDFImpl> &dims) {
      return boost::shared_ptr<HDFDataSet<HDFImpl> >
	(new HDFDataSet<HDFImpl>
	 (HDFImpl::template createDataSet<Type>(*group, path, dims)));
    }
    
    /**
     * Writes a 1D dataset stored in a std::vector to a dataset 
     * at the given path
     * Assumes the same layout in the file as in memory
     */
    template<typename Type>
    boost::shared_ptr<HDFDataSet<HDFImpl> >
    writeDataset(const boost::filesystem::path & path, const std::vector<Type> & data) {
      Slab<1,HDFImpl> dims(1,data.size());
      boost::shared_ptr<HDFDataSet<HDFImpl> > dataset = createDataset<1,Type>(path, dims);
      dataset->writeData(data);
      return dataset;
    }


    /**
     * Writes a dataset stored in data with memory layout specified
     * by memslab to a dataset 
     * at the given path with the fileformat given by fileslab
     */    
    template<int order, typename Type>
    boost::shared_ptr<HDFDataSet<HDFImpl> >
    writeDataset(const boost::filesystem::path & path, const Type* data, const Slab<order, HDFImpl> &memslab, const Slab<order, HDFImpl> &fileslab) {
      boost::shared_ptr<HDFDataSet<HDFImpl> > dataset = createDataset<Type>(path, fileslab);
      dataset->writeData(data, memslab);
      return dataset;
    }

  private:
    boost::shared_ptr<typename HDFImpl::group_type> group;
  };
}

#endif

