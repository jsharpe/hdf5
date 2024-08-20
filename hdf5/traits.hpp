
#pragma once

#include <boost/noncopyable.hpp>
#include <boost/fusion/support/is_sequence.hpp>
#include <boost/fusion/include/is_sequence.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/ref.hpp>
#include <boost/type_traits/remove_cv.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/count.hpp>
#include <boost/fusion/container/vector/convert.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/cstdint.hpp>

#include <string>
#include <memory>
#include <exception>
#include <vector>
#include <typeinfo>
#include <iostream>
#include <sstream>
#include <cassert>

#include <hdf5.h>
#ifdef H5_HAVE_PARALLEL
#include <mpi.h>
#endif

namespace hdf {
class FileOpenFailed : public std::exception {};
class DatasetExists : public std::exception {};
class DatasetOpenFailed : public std::exception {};
class DatasetWriteFailed : public std::exception {};
class AttributeExists : public std::exception {};
class AttributeCreateFailed : public std::exception {};
class AttributeWriteFailed : public std::exception {};
class AttributeOpenFailed : public std::exception {};
class GroupCreateFailed : public std::exception {};
class GroupOpenFailed : public std::exception {};
class GroupNotFound : public std::exception {};
class DatasetNotFound : public std::exception {};
class AttributeNotFound : public std::exception {};
class ChunkSizeDimMismatch : public std::exception {};

inline void output_dims(hid_t dataspace) {
    if(H5Sis_simple(dataspace)) {
        std::cout << "Num dims: " << H5Sget_simple_extent_ndims(dataspace) << "\n";
        std::vector<hsize_t> dims(H5Sget_simple_extent_ndims(dataspace));
        H5Sget_simple_extent_dims(dataspace, &dims[0], 0);
        for(std::size_t i = 0; i < dims.size(); ++i)
            std::cout << "[" << i << "] " << dims[i] << "\n";

        std::cout << "Selected " << H5Sget_select_elem_npoints(dataspace) << " points\n";
    }
}

namespace detail {

inline bool h5lexists(hid_t loc_id, const std::string& name) {
    std::size_t pos = name.find('/');
    while(pos != std::string::npos) {
        auto res = H5Lexists(loc_id, name.substr(pos).c_str(), H5P_DEFAULT);
        if(res < 0)
            throw;
        if(res == 0)
            return false;
        pos = name.find('/', pos + 1);
    }

    auto res = H5Lexists(loc_id, name.c_str(), H5P_DEFAULT);
    if(res < 0)
        throw;
    return (res > 0) ? true : false;
}

inline void check_errors() {
#if H5_VERS_MINOR >= 8
    H5Eprint(H5E_DEFAULT, NULL);
#else
    H5Eprint(NULL);
#endif
}

template <class T>
class wrapper {};

class Create {};
class ReadOnly {};

template <typename>
class data_type_traits;

template <class T>
size_t HDF5_datatype_size() {
    ///@todo! fix me for variable length datatypes/vectors/strings etc..
    size_t size = sizeof(T);
    return size;
}

template <class T, class Homogeneous>
struct TypeCreatorHelper {
    TypeCreatorHelper(hid_t& type) : type(type), offset(0) {
        type = H5Tcreate(H5T_COMPOUND, HDF5_datatype_size<T>());
        boost::mpl::for_each<typename boost::fusion::result_of::as_vector<T>::type>(boost::ref(*this));
        assert(HDF5_datatype_size<T>() == offset);
    }

    template <class T2>
    void operator()(T2&) const {
        data_type_traits<typename boost::remove_cv<T2>::type>::insert_data_type(type, offset);
    }

    hid_t& type;
    mutable size_t offset;
};

template <class T>
struct TypeCreatorHelper<T, boost::true_type> {
    TypeCreatorHelper(hid_t& type) { type = data_type_traits<typename boost::remove_cv<T>::type>::homogeneous_type(); }
};

struct DataTypeCreator {
    template <class T>
    DataTypeCreator(wrapper<T>) {
        TypeCreatorHelper<T, typename data_type_traits<typename boost::remove_cv<T>::type>::is_homogeneous> t(type);

        dim = data_type_traits<typename boost::remove_cv<T>::type>::dimsize();
    }

    template <class T>
    DataTypeCreator(wrapper<T*>) {
        TypeCreatorHelper<T, typename data_type_traits<typename boost::remove_cv<T>::type>::is_homogeneous> t(type);
        dim = data_type_traits<typename boost::remove_cv<T>::type>::dimsize();
    }
    hsize_t dim;
    hid_t type;
};

template <typename Type>
class data_type_traits {
public:
    typedef typename boost::fusion::result_of::as_vector<Type>::type vec_type;

    typedef typename boost::is_same<
        typename boost::fusion::result_of::size<vec_type>::type,
        boost::mpl::int_<boost::mpl::count<vec_type, typename boost::mpl::at_c<vec_type, 0>::type>::type::value> >::type
        is_homogeneous;

    static hid_t homogeneous_type() {
        return H5Tcopy(data_type_traits<typename boost::mpl::at_c<vec_type, 0>::type>::value());
    }

    static hsize_t dimsize() {
        if(is_homogeneous::value)
            return boost::mpl::count<vec_type, typename boost::mpl::at_c<vec_type, 0>::type>::type::value;
        else
            return 1;
    }

    static hsize_t num_type() {
        return boost::mpl::count<vec_type, typename boost::mpl::at_c<vec_type, 0>::type>::type::value;
    }

    static void insert_data_type(hid_t t, size_t& offset) {
        wrapper<Type> dt;
        DataTypeCreator d(dt);
        static int i = 0;
        std::string name(typeid(Type).name());
        std::stringstream ss;
        ss << i++;
        name += ss.str();
        H5Tinsert(t, name.c_str(), offset, d.type);
        offset += HDF5_datatype_size<Type>();
        H5Tclose(d.type);
    }
};

template <>
class data_type_traits<int> {
public:
    typedef boost::true_type is_homogeneous;
    static hid_t homogeneous_type() { return H5Tcopy(value()); }
    static hid_t value() { return H5T_NATIVE_INT; }
    static hsize_t dimsize() { return 1; }
    static hsize_t num_type() { return 1; }
    static void insert_data_type(hid_t t, size_t& offset) {
        static int i = 0;
        std::string name("int");
        std::stringstream ss;
        ss << i++;
        name += ss.str();
        H5Tinsert(t, name.c_str(), offset, value());
        offset += HDF5_datatype_size<int>();
    }
};

template <>
class data_type_traits<char> {
public:
    typedef boost::true_type is_homogeneous;
    static hid_t homogeneous_type() { return H5Tcopy(value()); }
    static hid_t value() { return H5T_NATIVE_CHAR; }
    static hsize_t dimsize() { return 1; }
    static hsize_t num_type() { return 1; }
    static void insert_data_type(hid_t t, size_t& offset) {
        static int i = 0;
        std::string name("char");
        std::stringstream ss;
        ss << i++;
        name += ss.str();
        H5Tinsert(t, name.c_str(), offset, value());
        offset += HDF5_datatype_size<char>();
    }
};

template <>
class data_type_traits<unsigned int> {
public:
    typedef boost::true_type is_homogeneous;
    static hid_t homogeneous_type() { return H5Tcopy(value()); }
    static hid_t value() { return H5T_NATIVE_UINT; }
    static hsize_t dimsize() { return 1; }
    static hsize_t num_type() { return 1; }
    static void insert_data_type(hid_t t, size_t& offset) {
        static int i = 0;
        std::string name("uint");
        std::stringstream ss;
        ss << i++;
        name += ss.str();
        H5Tinsert(t, name.c_str(), offset, value());
        offset += HDF5_datatype_size<unsigned int>();
    }
};

template <>
class data_type_traits<long> {
public:
    typedef boost::true_type is_homogeneous;
    static hid_t homogeneous_type() { return H5Tcopy(value()); }
    static hid_t value() { return H5T_NATIVE_LONG; }
    static hsize_t dimsize() { return 1; }
    static hsize_t num_type() { return 1; }
    static void insert_data_type(hid_t t, size_t& offset) {
        static int i = 0;
        std::string name("uint");
        std::stringstream ss;
        ss << i++;
        name += ss.str();
        H5Tinsert(t, name.c_str(), offset, value());
        offset += HDF5_datatype_size<long>();
    }
};

template <>
class data_type_traits<long long> {
public:
    typedef boost::true_type is_homogeneous;
    static hid_t homogeneous_type() { return H5Tcopy(value()); }
    static hid_t value() { return H5T_NATIVE_LLONG; }
    static hsize_t dimsize() { return 1; }
    static hsize_t num_type() { return 1; }
    static void insert_data_type(hid_t t, size_t& offset) {
        static int i = 0;
        std::string name("longlong");
        std::stringstream ss;
        ss << i++;
        name += ss.str();
        H5Tinsert(t, name.c_str(), offset, value());
        offset += HDF5_datatype_size<long long>();
    }
};

template <>
class data_type_traits<unsigned long> {
public:
    typedef boost::true_type is_homogeneous;
    static hid_t homogeneous_type() { return H5Tcopy(value()); }
    static hid_t value() { return H5T_NATIVE_ULONG; }
    static hsize_t dimsize() { return 1; }
    static hsize_t num_type() { return 1; }
    static void insert_data_type(hid_t t, size_t& offset) {
        static int i = 0;
        std::string name("uint");
        std::stringstream ss;
        ss << i++;
        name += ss.str();
        H5Tinsert(t, name.c_str(), offset, value());
        offset += HDF5_datatype_size<unsigned long>();
    }
};

template <>
class data_type_traits<boost::uint16_t> {
public:
    typedef boost::true_type is_homogeneous;
    static hid_t homogeneous_type() { return H5Tcopy(value()); }
    static hid_t value() { return H5T_NATIVE_UINT16; }
    static hsize_t dimsize() { return 1; }
    static hsize_t num_type() { return 1; }
    static void insert_data_type(hid_t t, size_t& offset) {
        static int i = 0;
        std::string name("uint16");
        std::stringstream ss;
        ss << i++;
        name += ss.str();
        H5Tinsert(t, name.c_str(), offset, value());
        offset += HDF5_datatype_size<boost::uint16_t>();
    }
};

template <>
class data_type_traits<boost::int16_t> {
public:
    typedef boost::true_type is_homogeneous;
    static hid_t homogeneous_type() { return H5Tcopy(value()); }
    static hid_t value() { return H5T_NATIVE_INT16; }
    static hsize_t dimsize() { return 1; }
    static hsize_t num_type() { return 1; }
    static void insert_data_type(hid_t t, size_t& offset) {
        static int i = 0;
        std::string name("int16");
        std::stringstream ss;
        ss << i++;
        name += ss.str();
        H5Tinsert(t, name.c_str(), offset, value());
        offset += HDF5_datatype_size<boost::int16_t>();
    }
};

template <>
class data_type_traits<float> {
public:
    typedef boost::true_type is_homogeneous;
    static hid_t homogeneous_type() { return H5Tcopy(value()); }
    static hid_t value() { return H5T_NATIVE_FLOAT; }
    static hsize_t dimsize() { return 1; }
    static hsize_t num_type() { return 1; }
    static void insert_data_type(hid_t t, size_t& offset) {
        static int i = 0;
        std::string name("float");
        std::stringstream ss;
        ss << i++;
        name += ss.str();
        H5Tinsert(t, name.c_str(), offset, value());
        offset += HDF5_datatype_size<float>();
    }
};

template <>
class data_type_traits<double> {
public:
    typedef boost::true_type is_homogeneous;
    static hid_t homogeneous_type() { return H5Tcopy(value()); }
    static hid_t value() { return H5T_NATIVE_DOUBLE; }
    static hsize_t dimsize() { return 1; }
    static hsize_t num_type() { return 1; }
    static void insert_data_type(hid_t t, size_t& offset) {
        static int i = 0;
        std::string name("double");
        std::stringstream ss;
        ss << i++;
        name += ss.str();
        H5Tinsert(t, name.c_str(), offset, value());
        offset += HDF5_datatype_size<double>();
    }
};

class HDF5FileHolder : boost::noncopyable {
public:
    HDF5FileHolder(const std::string& path) {
        hid_t plist_id;
        plist_id = H5Pcreate(H5P_FILE_ACCESS);

        if(H5Fis_hdf5(path.c_str()) > 0) {
            file = H5Fopen(path.c_str(), H5F_ACC_RDWR, plist_id);
        } else {
            file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
        }
        H5Pclose(plist_id);
        if(file < 0)
            throw FileOpenFailed();
        check_errors();
    }
    HDF5FileHolder(const std::string& path, ReadOnly) {
        hid_t plist_id;
        plist_id = H5Pcreate(H5P_FILE_ACCESS);

        if(H5Fis_hdf5(path.c_str()) > 0) {
            file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, plist_id);
        } else {
            H5Pclose(plist_id);
            throw FileOpenFailed();
        }
        H5Pclose(plist_id);
        if(file < 0) {
            throw FileOpenFailed();
        }
        check_errors();
    }
    HDF5FileHolder(const std::string& path, Create) {
        hid_t plist_id;
        plist_id = H5Pcreate(H5P_FILE_ACCESS);

        file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);

        H5Pclose(plist_id);
        if(file < 0) {
            throw FileOpenFailed();
        }

        check_errors();
    }
#ifdef H5_HAVE_PARALLEL
    HDF5FileHolder(const std::string& path, MPI_Comm comm) {
        hid_t plist_id;
        plist_id = H5Pcreate(H5P_FILE_ACCESS);

        // https://wickie.hlrs.de/platforms/index.php/MPI-IO

        if(H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL) < 0) {
            throw FileOpenFailed();
        }

        int rank;
        MPI_Comm_rank(comm, &rank);
        int8_t should_open = 0;
        if(rank == 0) {
            auto res = H5Fis_hdf5(path.c_str());
            // Not a HDF5 file format
            if(res == 0)
                throw FileOpenFailed();
            should_open = (res > 0) ? 1 : 0;
        }

        MPI_Bcast(&should_open, 1, MPI_INT8_T, 0, comm);

        if(should_open == 1) {
            file = H5Fopen(path.c_str(), H5F_ACC_RDWR, plist_id);
        } else {
            file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
        }
        H5Pclose(plist_id);
        if(file < 0) {
            throw FileOpenFailed();
        }
    }
    HDF5FileHolder(const std::string& path, MPI_Comm comm, ReadOnly) {
        hid_t plist_id;
        plist_id = H5Pcreate(H5P_FILE_ACCESS);

        // https://wickie.hlrs.de/platforms/index.php/MPI-IO

        if(H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL) < 0) {
            throw FileOpenFailed();
        }

        if(H5Fis_hdf5(path.c_str()) > 0) {
            file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, plist_id);
        } else {
            H5Pclose(plist_id);
            throw FileOpenFailed();
        }
        H5Pclose(plist_id);
        if(file < 0) {
            throw FileOpenFailed();
        }
    }

    HDF5FileHolder(const std::string& path, MPI_Comm comm, Create) {
        hid_t plist_id;
        plist_id = H5Pcreate(H5P_FILE_ACCESS);

        // https://wickie.hlrs.de/platforms/index.php/MPI-IO

        if(H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL) < 0) {
            throw FileOpenFailed();
        }

        file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);

        H5Pclose(plist_id);
        if(file < 0) {
            throw FileOpenFailed();
        }
    }
#endif

    ~HDF5FileHolder() {
        H5Fclose(file);
    }

    void flush() {
        H5Fflush(file, H5F_SCOPE_GLOBAL);
    }

    hid_t hid() const {
        return file;
    }

private:
    hid_t file;
};

class HDF5DataSpace {
public:
    HDF5DataSpace(hid_t space) : dataspace(space) {
        if(space < 0)
            throw;
        assert(H5Iget_type(space) == H5I_DATASPACE);
    }

    HDF5DataSpace(const HDF5DataSpace& other) {
        dataspace = H5Scopy(other.hid());
        check_errors();
    }

    HDF5DataSpace(const std::vector<hsize_t>& dims) {
        dataspace = H5Screate_simple(dims.size(), &dims[0], NULL);
        check_errors();
    }

    HDF5DataSpace(const std::vector<hsize_t>& dims, const std::vector<hsize_t>& maxdims) {
        assert(dims.size() == maxdims.size() || maxdims.empty());
        if(maxdims.empty())
            dataspace = H5Screate_simple(dims.size(), &dims[0], NULL);
        else
            dataspace = H5Screate_simple(dims.size(), &dims[0], &maxdims[0]);
        check_errors();
    }

    HDF5DataSpace(const HDF5DataSpace& orig, const std::vector<hsize_t>& offset, const std::vector<hsize_t>& stride,
                  const std::vector<hsize_t>& count) {
        dataspace = H5Scopy(orig.hid());
        if(stride.size()) {
            assert(offset.size() == stride.size() && stride.size() == count.size());

            H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, &offset[0], &stride[0], &count[0], NULL);
        } else
            H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, &offset[0], NULL, &count[0], NULL);

        check_errors();
    }

    HDF5DataSpace(const HDF5DataSpace& orig, const std::vector<hsize_t>& offset, const std::vector<hsize_t>& stride,
                  const std::vector<hsize_t>& count, const std::vector<hsize_t>& block) {
        assert(offset.size() == stride.size() && stride.size() == count.size() && count.size() == block.size());
        dataspace = H5Scopy(orig.hid());
        H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, &offset[0], &stride[0], &count[0], &block[0]);
        check_errors();
    }

    ~HDF5DataSpace() {
        H5Sclose(dataspace);
        check_errors();
    }

    hsize_t getNumDimensions() const { return H5Sget_simple_extent_ndims(dataspace); }

    std::vector<hsize_t> getDimensions() const {
        std::vector<hsize_t> dims(getNumDimensions());
        H5Sget_simple_extent_dims(dataspace, &dims[0], 0);
        return dims;
    }

    void selectAll() { H5Sselect_all(hid()); }

    void selectNone() { H5Sselect_none(hid()); }

    hid_t hid() const { return dataspace; }

    template <typename Type>
    static std::unique_ptr<HDF5DataSpace> selectSubset(const HDF5DataSpace& orig, const std::vector<Type>& mapping) {
        std::unique_ptr<HDF5DataSpace> newSpace(new HDF5DataSpace(orig));

        hsize_t dataspaceDims = newSpace->getNumDimensions();

        std::size_t numCoords = mapping.size() * dataspaceDims;
        std::vector<hsize_t> dims = newSpace->getDimensions();
        hsize_t dim2size = 1;
        if(dataspaceDims > 1) {
            numCoords *= dims[1];
            dim2size = dims[1];
        }
        std::size_t numElements = numCoords / dataspaceDims;

        newSpace->elements.resize(numCoords);
        auto j = newSpace->elements.begin();
        for(const auto& i : mapping) {
            for(hsize_t l = 0; l < dim2size; ++l) {
                *j = i;
                ++j;

                for(hsize_t k = 1; k < dataspaceDims; ++k, ++j)
                    *j = l;
            }

            // printf("%i %i\n",*i,*j);
        }
        /*
           for(int i=0;i<newSpace->elements.size()/dataspaceDims;++i)
           {
           printf(" %i %i \n", newSpace->elements[i*dataspaceDims],newSpace->elements[i*dataspaceDims+1]);
           }
         */
        H5Sselect_elements(newSpace->hid(), H5S_SELECT_SET, numElements, &newSpace->elements[0]);

        // int t = H5Sget_select_npoints(newSpace->hid());

        // printf(" dims %i %i %i %i %i %i\n", numElements,numCoords,dataspaceDims,dims[0],t,orig.getNumDimensions());

        return newSpace;
    }

public:
    /**
     * Logically and the two slabs together
     */
    HDF5DataSpace& operator&(const HDF5DataSpace& other) {
        checkExtentsMatch(other);
        return *this;
    }

    /**
     * Logically xor the two slabs together
     */
    HDF5DataSpace& operator^(const HDF5DataSpace& other) { return *this; }

    /**
     * Logically or the two slabs together
     */
    HDF5DataSpace& operator|(const HDF5DataSpace& other) { return *this; }

    /**
     * Logical not
     */
    HDF5DataSpace& operator!() { return *this; }

private:
    void checkExtentsMatch(const HDF5DataSpace& other) const {
#if H5_VERS_MINOR >= 8
        htri_t r = H5Sextent_equal(hid(), other.hid());
        if(r == 0)
            throw;  // Extents don't match
        if(r < 1)
            throw;  // Error
#endif
    }

private:
    hid_t dataspace;

    // Used for selections
    std::vector<hsize_t> elements;
};

class HDF5DataType : boost::noncopyable {
public:
    template <class Type>
    HDF5DataType(wrapper<Type> t) {
        createType(t);
    }

    template <class Type>
    HDF5DataType(Type) {
        wrapper<Type> t;
        createType(t);
    }

    ~HDF5DataType() {
        /*status = */ H5Tclose(datatype);
        check_errors();
    }

    void setLittleEndian() {
        H5Tset_order(datatype, H5T_ORDER_LE);
        check_errors();
    }

    void setBigEndian() {
        H5Tset_order(datatype, H5T_ORDER_BE);
        check_errors();
    }

    hsize_t getDim() const { return dim; }

    hid_t hid() const { return datatype; }

private:
    template <class Type>
    void createType(wrapper<Type> t) {
        DataTypeCreator type(t);
        datatype = type.type;
        dim = type.dim;
        check_errors();
    }

private:
    hsize_t dim;
    hid_t datatype;
};

class HDF5DataSet : boost::noncopyable {
public:
    template <class Parent>
    HDF5DataSet(Parent& p, const std::string& name) {
        if(h5lexists(p.hid(), name.c_str()) != true) {
            throw DatasetNotFound();
        }

#if H5_VERS_MINOR >= 8
        dataset = H5Dopen(p.hid(), name.c_str(), H5P_DEFAULT);
#else
        dataset = H5Dopen(p.hid(), name.c_str());
#endif
        if(dataset < 0) {
            throw DatasetOpenFailed();
        }
        space.reset(new HDF5DataSpace(H5Dget_space(dataset)));
    }

    template <class Parent>
    HDF5DataSet(Parent& p, const std::string name, const HDF5DataType& datatype, const HDF5DataSpace& dataspace,
                const std::vector<hsize_t> chunk_dims = std::vector<hsize_t>()) {
        hid_t cparms;
        if(h5lexists(p.hid(), name.c_str()) == true) {
            throw DatasetExists();
        }

        cparms = H5Pcreate(H5P_DATASET_CREATE);
        // Chunk the datset
        if(chunk_dims.size()) {
            if(dataspace.getNumDimensions() == chunk_dims.size()) {
                //       hsize_t chunk_dims[2] = {2,2};
                //       H5Pset_chunk(cparms, 2, chunk_dims);
                H5Pset_chunk(cparms, chunk_dims.size(), &chunk_dims[0]);
            } else {
                H5Pclose(cparms);
                throw ChunkSizeDimMismatch();
            }
        }

#if H5_VERS_MINOR >= 8
        dataset = H5Dcreate(p.hid(), name.c_str(), datatype.hid(), dataspace.hid(), H5P_DEFAULT, cparms, H5P_DEFAULT);
#else
        dataset = H5Dcreate(p.hid(), name.c_str(), datatype.hid(), dataspace.hid(), cparms);
#endif
        H5Pclose(cparms);

        if(dataset < 0) {
            throw DatasetOpenFailed();
        }
        space.reset(new HDF5DataSpace(H5Dget_space(dataset)));
    }

    ~HDF5DataSet() {
        /*status = */ H5Dclose(dataset);
        check_errors();
    }

    const std::shared_ptr<HDF5DataSpace>& getDataSpace() const {
        return space;
    }

    template <typename Type>
    std::unique_ptr<HDF5DataSet> selectSubset(const std::vector<Type>& mapping) {
        std::unique_ptr<HDF5DataSet> newDataset(new HDF5DataSet(dataset));
        newDataset->space = HDF5DataSpace::selectSubset(*space, mapping);
        return newDataset;
    }

    ///@todo: getAttribute(const char * name)

    hid_t hid() const {
        return dataset;
    }

private:
    HDF5DataSet(hid_t d) : dataset(d) {
        if(dataset < 0)
            throw;
        assert(H5Iget_type(dataset) == H5I_DATASET);
        H5Iinc_ref(dataset);
    }

private:
    hid_t dataset;
    std::shared_ptr<HDF5DataSpace> space;
};

class HDF5Group : boost::noncopyable {
public:
    template <class Parent>
    HDF5Group(Parent& p, const std::string& path, Create) {
#if H5_VERS_MINOR >= 8
        group = H5Gcreate(p.hid(), path.c_str(), 0, H5P_DEFAULT, H5P_DEFAULT);
#else
        group = H5Gcreate(p.hid(), path.c_str(), 0);
#endif
        if(group < 0)
            throw GroupCreateFailed();
        check_errors();
    }

    template <class Parent>
    HDF5Group(Parent& p, const std::string& path, bool create) {
        if(path == "/" || h5lexists(p.hid(), path.c_str())) {
#if H5_VERS_MINOR >= 8
            group = H5Gopen(p.hid(), path.c_str(), H5P_DEFAULT);
#else
            group = H5Gopen(p.hid(), path.c_str());
#endif
            if(group < 0) {
                throw GroupOpenFailed();
            }
        } else if(create) {
            // Group didn't exist and we've asked to create the group
#if H5_VERS_MINOR >= 8
            group = H5Gcreate(p.hid(), path.c_str(), 0, H5P_DEFAULT, H5P_DEFAULT);
#else
            group = H5Gcreate(p.hid(), path.c_str(), 0);
#endif
            if(group < 0)
                throw GroupCreateFailed();  // Error creating group
        } else {
            throw GroupNotFound();
        }
    }

    /**
     * Create external link
     */
    template <class Parent>
    HDF5Group(Parent& p, const std::string& externalFile, const std::string& externalPath, const std::string& path,
              Create) {
#if H5_VERS_MINOR >= 8
        group = H5Lcreate_external(externalFile.c_str(), externalPath.c_str(), p.hid(), path.c_str(), H5P_DEFAULT,
                                   H5P_DEFAULT);
#else
        group = H5Lcreate_external(p.hid(), path.c_str());
#endif
        if(group < 0)
            throw;
        check_errors();
    }

    ~HDF5Group() {
        H5Gclose(group);
        check_errors();
    }

    hid_t hid() const {
        return group;
    }

private:
    hid_t group;
};

class HDF5Attribute : boost::noncopyable {
public:
    template <class Object>
    HDF5Attribute(Object& p, const std::string& name) {
        if(H5Aexists(p.hid(), name.c_str()) != true) {
            throw AttributeNotFound();
        }

        attribute = H5Aopen_name(p.hid(), name.c_str());
        if(attribute < 0) {
            throw AttributeOpenFailed();
        }
    }

    template <class Type, class Object>
    HDF5Attribute(Object& p, const std::string& name, const std::vector<hsize_t>& dims,
                  const std::vector<hsize_t>& maxdims, Type t) {
        if(H5Aexists(p.hid(), name.c_str()) == true) {
            throw AttributeExists();
        }

        HDF5DataSpace space(dims, maxdims);
        HDF5DataType type(t);

#if H5_VERS_MINOR >= 8
        attribute = H5Acreate(p.hid(), name.c_str(), type.hid(), space.hid(), H5P_DEFAULT, H5P_DEFAULT);
#else
        attribute = H5Acreate(p.hid(), name.c_str(), type.hid(), space.hid(), H5P_DEFAULT);
#endif
        if(attribute < 0) {
            throw AttributeCreateFailed();
        }
    }

    ~HDF5Attribute() {
        H5Aclose(attribute);
    }

    hid_t hid() const {
        return attribute;
    }

private:
    hid_t attribute;
};

}  // namespace detail

class HDF5Traits {
public:
    typedef detail::HDF5FileHolder file_handle_type;
    typedef detail::HDF5Group group_type;
    typedef detail::HDF5DataType datatype_type;
    typedef detail::HDF5DataSet dataset_type;
    typedef detail::HDF5Attribute attribute_type;
    typedef detail::HDF5DataSpace slab_type;

    static std::unique_ptr<file_handle_type> open(const std::string& path, bool truncate, bool readonly) {
        if(readonly) {
            detail::ReadOnly ro;
            return std::unique_ptr<file_handle_type>(new file_handle_type(path, ro));
        }
        if(truncate) {
            detail::Create c;
            return std::unique_ptr<file_handle_type>(new file_handle_type(path, c));
        }
        return std::unique_ptr<file_handle_type>(new file_handle_type(path));
    }
#ifdef H5_HAVE_PARALLEL
    static std::unique_ptr<file_handle_type> parallel_open(const std::string& path, bool truncate, bool readonly,
                                                           MPI_Comm comm) {
        if(readonly) {
            detail::ReadOnly ro;
            return std::unique_ptr<file_handle_type>(new file_handle_type(path, comm, ro));
        }
        if(truncate) {
            detail::Create c;
            return std::unique_ptr<file_handle_type>(new file_handle_type(path, comm, c));
        }
        return std::unique_ptr<file_handle_type>(new file_handle_type(path, comm));
    }
#endif
    template <typename FileHandle>
    static std::unique_ptr<group_type> openGroup(FileHandle& f, const std::string& path, bool create) {
        return std::unique_ptr<group_type>(new group_type(f, path, create));
    }

    static std::unique_ptr<group_type> openGroup(group_type& f, const std::string& path, bool create) {
        return std::unique_ptr<group_type>(new group_type(f, path, create));
    }

    template <typename FileHandle>
    static std::unique_ptr<group_type> createGroup(FileHandle& f, const std::string& path) {
        return std::unique_ptr<group_type>(new group_type(f, path, detail::Create()));
    }

    static std::unique_ptr<group_type> createGroup(group_type& f, const std::string& path) {
        return std::unique_ptr<group_type>(new group_type(f, path, detail::Create()));
    }

    static std::unique_ptr<group_type> createExternalLink(group_type& f, const std::string& externalFile,
                                                          const std::string& externalPath, const std::string& path) {
        return std::unique_ptr<group_type>(new group_type(f, externalFile, externalPath, path, detail::Create()));
    }

    template <typename FileHandle>
    static std::unique_ptr<dataset_type> openDataSet(FileHandle& f, const std::string& path) {
        return std::unique_ptr<dataset_type>(new dataset_type(f, path));
    }

    static std::unique_ptr<dataset_type> openDataSet(group_type& f, const std::string& path) {
        return std::unique_ptr<dataset_type>(new dataset_type(f, path));
    }

    template <class Type>
    static std::unique_ptr<attribute_type> createAttribute(group_type& g, const std::string& name,
                                                           const std::vector<hsize_t>& dims,
                                                           const std::vector<hsize_t>& maxdims) {
        return std::unique_ptr<attribute_type>(new attribute_type(g, name, dims, maxdims, Type()));
    }

    template <class Type>
    static std::unique_ptr<attribute_type> createAttribute(dataset_type& g, const std::string& name,
                                                           const std::vector<hsize_t>& dims,
                                                           const std::vector<hsize_t>& maxdims) {
        return std::unique_ptr<attribute_type>(new attribute_type(g, name, dims, maxdims, Type()));
    }

    static std::unique_ptr<attribute_type> openAttribute(group_type& g, const std::string& name) {
        return std::unique_ptr<attribute_type>(new attribute_type(g, name));
    }

    static std::unique_ptr<attribute_type> openAttribute(dataset_type& g, const std::string& name) {
        return std::unique_ptr<attribute_type>(new attribute_type(g, name));
    }

    template <class Type, typename FileHandle>
    static std::unique_ptr<dataset_type> createDataSet(FileHandle& f, const std::string& path,
                                                       const detail::HDF5DataSpace& space) {
        detail::HDF5DataType type((detail::wrapper<Type>()));
        return std::unique_ptr<dataset_type>(new dataset_type(f, path, type, space));
    }

    template <typename Type>
    static std::unique_ptr<dataset_type> createDataSet(group_type& f, const std::string& path,
                                                       const detail::HDF5DataSpace& space,
                                                       const std::vector<hsize_t> chunk_dims = std::vector<hsize_t>()) {
        detail::wrapper<Type> t;
        detail::HDF5DataType datatype(t);
        // Note DataSpace assumes that datatype is of dim 1
        // This is incorrect for homogeneous complex types
        // We should create new dataspace to account for dimension
        if(datatype.getDim() > 1) {
            std::vector<hsize_t> dims(2, space.getDimensions()[0]);
            dims[1] = datatype.getDim();
            detail::HDF5DataSpace filespace(dims);
            return std::unique_ptr<dataset_type>(new dataset_type(f, path, datatype, filespace, chunk_dims));
        } else {
            return std::unique_ptr<dataset_type>(new dataset_type(f, path, datatype, space, chunk_dims));
        }
    }

    template <typename Type>
    static void write_attribute(const attribute_type& attribute, const Type& data) {
        detail::wrapper<Type> t;
        detail::HDF5DataType datatype(t);
        auto err = H5Awrite(attribute.hid(), datatype.hid(), &data);
        if(err < 0) {
            throw AttributeWriteFailed();
        }
    }

    template <typename Type>
    static void write_attribute(const attribute_type& attribute, const std::vector<Type>& data) {
        detail::wrapper<Type> t;
        detail::HDF5DataType memdatatype(t);
        auto err = H5Awrite(attribute.hid(), memdatatype.hid(), &data);
        if(err < 0) {
            throw AttributeWriteFailed();
        }
    }

    /*    struct T{
          int a; float b;
          };
     */
    template <typename Type>
    static void write_dataset(const dataset_type& dataset, const std::vector<Type>& data) {
        std::vector<hsize_t> d = getDims(data);
        detail::HDF5DataSpace memorySpace(d);
        detail::wrapper<Type> t;
        detail::HDF5DataType memdatatype(t);
        /*
           if(d.size()==1){
           T tt[2];
           tt[0].a = 1; tt[0].b = 2;
           tt[1].a = 3; tt[1].b = 4;



           herr_t status = H5Dwrite(dataset.hid(), memdatatype.hid(), memorySpace.hid(), H5S_ALL,
           H5P_DEFAULT, &data[0]);
           assert(status == 0);
           }
           else
         */
        {
            herr_t status =
                H5Dwrite(dataset.hid(), memdatatype.hid(), memorySpace.hid(), H5S_ALL, H5P_DEFAULT, &data[0]);
            if(status < 0) {
                throw DatasetWriteFailed();
            }
        }
    }
#ifdef H5_HAVE_PARALLEL
    template <typename Type>
    static void write_parallel_dataset(const dataset_type& dataset, const std::vector<Type>& data) {
        hid_t plist_id;
        plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

        std::vector<hsize_t> d = getDims(data);
        detail::HDF5DataSpace memorySpace(d);
        detail::wrapper<Type> t;
        detail::HDF5DataType memdatatype(t);
        auto err = H5Dwrite(dataset.hid(), memdatatype.hid(), memorySpace.hid(), H5S_ALL, plist_id, &data[0]);
        if(err < 0) {
            throw DatasetWriteFailed();
        }
    }
#endif
    template <typename Type>
    static void write_dataset(const dataset_type& dataset, const Type* data, const detail::HDF5DataSpace& memorySpace) {
        detail::wrapper<Type> t;
        detail::HDF5DataType memdatatype(t);
        auto fileSpace = dataset.getDataSpace();

        auto err = H5Dwrite(dataset.hid(), memdatatype.hid(), memorySpace.hid(), fileSpace->hid(), H5P_DEFAULT, data);
        // H5Sclose(fileSpace);
        if(err < 0) {
            throw DatasetWriteFailed();
        }
    }
#ifdef H5_HAVE_PARALLEL
    template <typename Type>
    static void write_parallel_dataset(const dataset_type& dataset, const Type* data,
                                       const detail::HDF5DataSpace& memorySpace) {
        hid_t plist_id;
        plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

        detail::wrapper<Type> t;
        detail::HDF5DataType memdatatype(t);
        auto fileSpace = dataset.getDataSpace();

        auto err = H5Dwrite(dataset.hid(), memdatatype.hid(), memorySpace.hid(), fileSpace->hid(), plist_id, data);
        // H5Sclose(fileSpace);
        if(err < 0) {
            throw DatasetWriteFailed();
        }
    }
#endif
    template <typename Type>
    static void write_dataset(const dataset_type& dataset, const Type* data, const detail::HDF5DataSpace& memorySpace,
                              const detail::HDF5DataSpace& fileSpace) {
        detail::wrapper<Type> t;
        detail::HDF5DataType memdatatype(t);
        // hid_t fileSpace = H5Dget_space(dataset.hid());
        auto err = H5Dwrite(dataset.hid(), memdatatype.hid(), memorySpace.hid(), fileSpace.hid(), H5P_DEFAULT, data);
        // H5Sclose(fileSpace);
        if(err < 0) {
            throw DatasetWriteFailed();
        }
    }
#ifdef H5_HAVE_PARALLEL
    template <typename Type>
    static void write_parallel_dataset(const dataset_type& dataset, const Type* data,
                                       const detail::HDF5DataSpace& memorySpace,
                                       const detail::HDF5DataSpace& fileSpace) {
        hid_t plist_id;
        plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

        detail::wrapper<Type> t;
        detail::HDF5DataType memdatatype(t);
        // hid_t fileSpace = H5Dget_space(dataset.hid());
        auto err = H5Dwrite(dataset.hid(), memdatatype.hid(), memorySpace.hid(), fileSpace.hid(), plist_id, data);
        // H5Sclose(fileSpace);
        if(err < 0) {
            throw DatasetWriteFailed();
        }
    }
#endif
    template <typename Type>
    static void read_attribute(const attribute_type& attribute, Type& data) {
        detail::wrapper<Type> t;
        detail::HDF5DataType datatype(t);
        H5Aread(attribute.hid(), datatype.hid(), &data);
    }

    template <typename Type>
    static void read_dataset(const dataset_type& dataset, std::vector<Type>& data) {
        auto fileSpace = dataset.getDataSpace();
        if(data.empty()) {
            hsize_t sel = H5Sget_select_npoints(fileSpace->hid());

            /*
               std::vector<hsize_t> dims = fileSpace->getDimensions();
               std::size_t size = dims[0];
               for (int i = 1; i < dims.size(); ++i)
               size *= dims[i];
               assert(sel != size);
             */
            data.resize(sel);
        }
        std::vector<hsize_t> d(1, data.size());
        detail::HDF5DataSpace memorySpace(d);
        detail::wrapper<Type> t;
        detail::HDF5DataType datatype(t);
        H5Dread(dataset.hid(), datatype.hid(), memorySpace.hid(), fileSpace->hid(), H5P_DEFAULT, &data[0]);
    }

    template <typename Type>
    static void read_dataset(const dataset_type& dataset, Type* data, const detail::HDF5DataSpace& memorySpace) {
        auto fileSpace = dataset.getDataSpace();
        detail::wrapper<Type> t;
        detail::HDF5DataType datatype(t);

        //       output_dims(memorySpace.hid());
        //       output_dims(fileSpace->hid());
        H5Dread(dataset.hid(), datatype.hid(), memorySpace.hid(), fileSpace->hid(), H5P_DEFAULT, data);
    }

    template <typename Type>
    static void read_dataset(const dataset_type& dataset, Type* data, const detail::HDF5DataSpace& memorySpace,
                             const detail::HDF5DataSpace& fileSpace) {
        detail::wrapper<Type> t;
        detail::HDF5DataType datatype(t);

        //       output_dims(memorySpace.hid());
        //       output_dims(fileSpace->hid());
        H5Dread(dataset.hid(), datatype.hid(), memorySpace.hid(), fileSpace.hid(), H5P_DEFAULT, data);
    }
#ifdef H5_HAVE_PARALLEL
    template <typename Type>
    static void read_parallel_dataset(const dataset_type& dataset, Type* data,
                                      const detail::HDF5DataSpace& memorySpace) {
        hid_t plist_id;
        plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

        auto fileSpace = dataset.getDataSpace();
        detail::wrapper<Type> t;
        detail::HDF5DataType datatype(t);

        //       output_dims(memorySpace.hid());
        //       output_dims(fileSpace->hid());
        H5Dread(dataset.hid(), datatype.hid(), memorySpace.hid(), fileSpace->hid(), plist_id, data);
    }

    template <typename Type>
    static void read_parallel_dataset(const dataset_type& dataset, Type* data, const detail::HDF5DataSpace& memorySpace,
                                      const detail::HDF5DataSpace& fileSpace) {
        hid_t plist_id;
        plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

        detail::wrapper<Type> t;
        detail::HDF5DataType datatype(t);

        // output_dims(memorySpace.hid());
        // output_dims(fileSpace.hid());
        H5Dread(dataset.hid(), datatype.hid(), memorySpace.hid(), fileSpace.hid(), plist_id, data);
    }
#endif
    template <typename Type>
    static void read_dataset(const dataset_type& dataset, Type& data) {
        auto fileSpace = dataset.getDataSpace();
        detail::wrapper<Type> t;
        detail::HDF5DataType datatype(t);
        H5Dread(dataset.hid(), datatype.hid(), fileSpace->hid(), fileSpace->hid(), H5P_DEFAULT, data);
    }

    template <typename HID>
    static void deleteDataset(HID& h, const std::string& path) {
        H5Ldelete(h.hid(), path.c_str(), H5P_DEFAULT);
    }

    template <typename Type>
    static std::vector<hsize_t> getDims(const std::vector<Type>& t) {
        // if( detail::data_type_traits<Type>::is_homogeneous::value)
        {
            std::vector<hsize_t> dims(2, detail::data_type_traits<Type>::dimsize());
            dims[0] = t.size();
            return dims;
        } /*
                                             else
                                             {
                                             std::vector<hsize_t> dims(1);//2,
             detail::data_type_traits<Type>::num_type()); dims[0] = t.size(); return dims;
                                             }*/
    }
};
}  // namespace hdf
