#include <boost/filesystem/path.hpp>
#include <hdf5/hdfgroup.hpp>
#include <hdf5/hdf5/traits.hpp>

namespace hdf {
  template<class HDFImpl=HDF5Traits>
  class HDFFile : public HDFGroup<HDFImpl> {
  public:
    enum Flags {
      none = 0,
      truncate = 1
    };

    HDFFile(const boost::filesystem::path & path, Flags flags = none) {
      file = HDFImpl::open(path, flags==truncate);
      initFileGroup(*file);
    };
    
    ~HDFFile() {}
    
  private:
    boost::shared_ptr<typename HDFImpl::file_handle_type> file;
  };
}
