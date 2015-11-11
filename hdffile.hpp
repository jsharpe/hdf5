#ifndef hdffileH
#define hdffileH

#include <hdf5/hdfgroup.hpp>
#include <hdf5/hdf5/traits.hpp>
#include <string>

namespace hdf {
template<class HDFImpl=HDF5Traits>
class HDFFile : public HDFGroup<HDFImpl> {
  public:
    typedef HDFGroup<HDFImpl> HDFGroup_t;
    enum Flags {
        none = 0,
        truncate = 1
    };


    /**
     * Open the hdf5 file at the given location
     * Truncates the file if flags == truncate
     */
    HDFFile(const std::string & path, Flags flags = none) {
        file = HDFImpl::open(path, flags==truncate);
        HDFGroup_t::initFileGroup(*file);
    };

    ~HDFFile() {
        HDFGroup<HDFImpl>::group.reset();
    }

  private:
    std::shared_ptr<typename HDFImpl::file_handle_type> file;
};

#ifdef H5_HAVE_PARALLEL
template<class HDFImpl=HDF5Traits>
class HDFParallelFile : public HDFGroup<HDFImpl> {
  public:
    typedef HDFGroup<HDFImpl> HDFGroup_t;
    enum Flags {
        none = 0,
        truncate = 1
    };


    /**
     * Open the hdf5 file at the given location
     * Truncates the file if flags == truncate
     */
    HDFParallelFile(const std::string & path, Flags flags = none) {
        file = HDFImpl::parallel_open(path, flags==truncate);
        HDFGroup_t::initFileGroup(*file);
    };

    ~HDFParallelFile() {
        HDFGroup<HDFImpl>::group.reset();
    }

  private:
    std::shared_ptr<typename HDFImpl::parallel_file_handle_type> file;
};
#endif
}

#endif

