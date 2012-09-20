#ifndef hdfslabH
#define hdfslabH

#include <hdf5/hdf5/traits.hpp>
#include <vector>

namespace hdf {

  template<int order, class HDFImpl=HDF5Traits>
  class Slab
    : public HDFImpl::slab_type
  {
  public:
    Slab(const std::vector<hsize_t> &dims)
      : HDFImpl::slab_type(dims)
    {
    }

    Slab(const std::vector<hsize_t> &dims, const std::vector<hsize_t> &maxdims)
      : HDFImpl::slab_type(dims,maxdims)
    {
    }

    Slab(const Slab<order, HDFImpl> & orig,
	 const std::vector<hsize_t> & offset,
	 const std::vector<hsize_t> & stride,
	 const std::vector<hsize_t> & count)
      : HDFImpl::slab_type(orig, offset, stride, count)
    {}

    Slab(const Slab<order, HDFImpl> & orig,
	 const std::vector<hsize_t> & offset,
	 const std::vector<hsize_t> & stride,
	 const std::vector<hsize_t> & count,
	 const std::vector<hsize_t> & block)
      : HDFImpl::slab_type(orig, offset, stride, count, block)
    {}

  };
}

#endif
