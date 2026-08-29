#pragma once

#include <string>
#include <cstddef>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

class FileMapper {
public:
    FileMapper(const std::string& path) {
#ifdef _WIN32
        file_handle_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle_ == INVALID_HANDLE_VALUE) {
            std::cerr << "Failed to open file (Windows): " << path << std::endl;
            return;
        }

        LARGE_INTEGER size;
        if (!GetFileSizeEx(file_handle_, &size)) {
            std::cerr << "Failed to get file size (Windows): " << path << std::endl;
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            return;
        }
        file_size_ = static_cast<size_t>(size.QuadPart);

        mapping_handle_ = CreateFileMapping(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_handle_ == nullptr) {
            std::cerr << "Failed to create file mapping (Windows): " << path << std::endl;
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            return;
        }

        mapped_view_ = static_cast<const std::byte*>(MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0));
        if (mapped_view_ == nullptr) {
            std::cerr << "Failed to map view of file (Windows): " << path << std::endl;
            CloseHandle(mapping_handle_);
            CloseHandle(file_handle_);
            mapping_handle_ = nullptr;
            file_handle_ = INVALID_HANDLE_VALUE;
            return;
        }
#else
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ == -1) {
            std::cerr << "Failed to open file: " << path << std::endl;
            return;
        }

        struct stat st;
        if (fstat(fd_, &st) == -1) {
            std::cerr << "Failed to fstat file: " << path << std::endl;
            close(fd_);
            fd_ = -1;
            return;
        }
        file_size_ = st.st_size;

        mapped_view_ = static_cast<const std::byte*>(mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0));
        if (mapped_view_ == MAP_FAILED) {
            std::cerr << "Failed to mmap file: " << path << std::endl;
            close(fd_);
            fd_ = -1;
            mapped_view_ = nullptr;
            return;
        }
#endif
        is_open_ = (mapped_view_ != nullptr);
    }

    ~FileMapper() {
#ifdef _WIN32
        if (mapped_view_) {
            UnmapViewOfFile(mapped_view_);
        }
        if (mapping_handle_) {
            CloseHandle(mapping_handle_);
        }
        if (file_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle_);
        }
#else
        if (mapped_view_) {
            munmap(const_cast<void*>(static_cast<const void*>(mapped_view_)), file_size_);
        }
        if (fd_ != -1) {
            close(fd_);
        }
#endif
    }

    FileMapper(const FileMapper&) = delete;
    FileMapper& operator=(const FileMapper&) = delete;

    bool is_open() const { return is_open_; }
    const std::byte* data() const { return mapped_view_; }
    size_t size() const { return file_size_; }

private:
    bool is_open_ = false;
    const std::byte* mapped_view_ = nullptr;
    size_t file_size_ = 0;

#ifdef _WIN32
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};
