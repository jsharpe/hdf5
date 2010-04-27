#ifndef hdffileH
#define hdffileH

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

    /**
     * Open the hdf5 file at the given location
     * Truncates the file if flags == truncate
     */
    HDFFile(const boost::filesystem::path & path, Flags flags = none) {
      file = HDFImpl::open(path, flags==truncate);
      initFileGroup(*file);
    };

    ~HDFFile() {
      HDFGroup<HDFImpl>::group.reset();
    }

  private:
    boost::shared_ptr<typename HDFImpl::file_handle_type> file;
  };
}

#endif

