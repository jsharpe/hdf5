#ifndef hdfattributeH
#define hdfattributeH

#include <hdf5/hdf5/traits.hpp>
#include <boost/shared_ptr.hpp>

namespace hdf {
template<class HDFImpl=HDF5Traits>
class HDFAttribute {
  public:
    HDFAttribute(std::shared_ptr<typename HDFImpl::attribute_type> attribute)
        :attribute(attribute) {
    }

    template<typename Type>
    void
    writeData(const Type & data) {
        HDFImpl::write_attribute(*attribute, data);
    }

    template<typename Type>
    void
    readData(Type & data) {
        HDFImpl::read_attribute(*attribute, data);
    }

  private:
    std::shared_ptr<typename HDFImpl::attribute_type> attribute;
};
}

#endif
