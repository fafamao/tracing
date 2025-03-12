#ifndef MEM_POOL_H_
#define MEM_POOL_H_

#include <sys/resource.h>
#include <unistd.h>
#include <iostream>
#include <new>

class MemoryPool
{
public:
    MemoryPool(size_t pool_size)
    {
        _ptr = new (std::nothrow) char[pool_size];
        _allocated = 0;
        _remain = pool_size;
        if (_ptr == nullptr)
        {
            getMemoryInfo();
            std::cerr << "Memory allocation failed." << std::endl;
            abort();
        }
    };

    char *allocate(size_t buff_size)
    {
        if (buff_size > _remain)
        {
            std::cerr << "Allocated: " << buff_size << ". Pool size: " << _remain << std::endl;
            abort();
        }
        char *allocated_ptr = _ptr;
        _ptr += buff_size;
        return allocated_ptr;
    };

    ~MemoryPool()
    {
        delete[] _ptr;
    };

    void getMemoryInfo()
    {
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        total_ram = pages * page_size;

        std::cout << "Total RAM:" << total_ram << " BYTE" << std::endl;
    };

private:
    size_t total_ram;
    size_t _allocated;
    size_t _remain;
    char *_ptr;
};

#endif // MEM_POOL_H_