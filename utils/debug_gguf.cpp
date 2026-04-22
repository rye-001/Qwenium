/**
 * g++ -std=c++14 -o debug_gguf debug_gguf.cpp
 * ./debug_gguf
 */

 #include <iostream>
#include <fstream>
#include <vector>
#include <string>

static std::string get_model_path() {
    const char* path = std::getenv("QWEN_MODEL_PATH");
    return path ? std::string(path) : "";
}
struct GGUFHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
};

enum class GGUFValueType : uint32_t {
    UINT8   = 0, INT8    = 1, UINT16  = 2, INT16   = 3,
    UINT32  = 4, INT32   = 5, FLOAT32 = 6, BOOL    = 7,
    STRING  = 8, ARRAY   = 9, UINT64  = 10, INT64  = 11,
    FLOAT64 = 12,
};

bool skip_value(std::ifstream& file, GGUFValueType type) {
    switch (type) {
        case GGUFValueType::UINT8: return file.seekg(sizeof(uint8_t), std::ios::cur).good();
        case GGUFValueType::INT8: return file.seekg(sizeof(int8_t), std::ios::cur).good();
        case GGUFValueType::UINT16: return file.seekg(sizeof(uint16_t), std::ios::cur).good();
        case GGUFValueType::INT16: return file.seekg(sizeof(int16_t), std::ios::cur).good();
        case GGUFValueType::INT32: return file.seekg(sizeof(int32_t), std::ios::cur).good();
        case GGUFValueType::FLOAT32: return file.seekg(sizeof(float), std::ios::cur).good();
        case GGUFValueType::BOOL: return file.seekg(sizeof(bool), std::ios::cur).good();
        case GGUFValueType::UINT64: return file.seekg(sizeof(uint64_t), std::ios::cur).good();
        case GGUFValueType::INT64: return file.seekg(sizeof(int64_t), std::ios::cur).good();
        case GGUFValueType::FLOAT64: return file.seekg(sizeof(double), std::ios::cur).good();
        case GGUFValueType::STRING: {
            uint64_t value_len;
            if (!file.read(reinterpret_cast<char*>(&value_len), sizeof(value_len))) return false;
            return file.seekg(value_len, std::ios::cur).good();
        }
        case GGUFValueType::ARRAY: {
            GGUFValueType array_type;
            uint64_t array_len;
            if (!file.read(reinterpret_cast<char*>(&array_type), sizeof(array_type))) return false;
            if (!file.read(reinterpret_cast<char*>(&array_len), sizeof(array_len))) return false;
            
            if (array_type == GGUFValueType::STRING) {
                for (uint64_t j = 0; j < array_len; ++j) {
                    uint64_t str_len;
                    if (!file.read(reinterpret_cast<char*>(&str_len), sizeof(str_len))) return false;
                    if (!file.seekg(str_len, std::ios::cur).good()) return false;
                }
            } else {
                size_t element_size = 0;
                switch (array_type) {
                    case GGUFValueType::UINT8: element_size = sizeof(uint8_t); break;
                    case GGUFValueType::INT8: element_size = sizeof(int8_t); break;
                    case GGUFValueType::UINT16: element_size = sizeof(uint16_t); break;
                    case GGUFValueType::INT16: element_size = sizeof(int16_t); break;
                    case GGUFValueType::UINT32: element_size = sizeof(uint32_t); break;
                    case GGUFValueType::INT32: element_size = sizeof(int32_t); break;
                    case GGUFValueType::FLOAT32: element_size = sizeof(float); break;
                    case GGUFValueType::BOOL: element_size = sizeof(bool); break;
                    case GGUFValueType::UINT64: element_size = sizeof(uint64_t); break;
                    case GGUFValueType::INT64: element_size = sizeof(int64_t); break;
                    case GGUFValueType::FLOAT64: element_size = sizeof(double); break;
                    default: return false;
                }
                if (element_size > 0) {
                    return file.seekg(array_len * element_size, std::ios::cur).good();
                }
            }
            return true;
        }
        default:
            return false;
    }
}

int main() {
    std::string modelPath = get_model_path();
    std::ifstream file(modelPath, std::ios::binary); // Qwen3-0.6B-Q8_0.gguf
    if (!file) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    GGUFHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    std::cout << "Header: magic=0x" << std::hex << header.magic << ", version=" << std::dec << header.version 
              << ", tensors=" << header.tensor_count << ", metadata=" << header.metadata_kv_count << std::endl;

    for (uint64_t i = 0; i < header.metadata_kv_count; ++i) {
        uint64_t key_len;
        file.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
        
        std::vector<char> key_data(key_len + 1);
        file.read(key_data.data(), key_len);
        key_data[key_len] = '\0';
        
        GGUFValueType type;
        file.read(reinterpret_cast<char*>(&type), sizeof(type));
        
        std::string key_str(key_data.data());
        std::cout << "Key[" << i << "]: '" << key_str << "' (type=" << static_cast<uint32_t>(type) << ")";
        
        // Read and display value based on type
        switch (type) {
            case GGUFValueType::UINT32: {
                uint32_t value;
                file.read(reinterpret_cast<char*>(&value), sizeof(value));
                std::cout << " = " << value;
                break;
            }
            case GGUFValueType::FLOAT32: {
                float value;
                file.read(reinterpret_cast<char*>(&value), sizeof(value));
                std::cout << " = " << value;
                break;
            }
            case GGUFValueType::STRING: {
                uint64_t value_len;
                file.read(reinterpret_cast<char*>(&value_len), sizeof(value_len));
                std::vector<char> value_data(value_len + 1);
                file.read(value_data.data(), value_len);
                value_data[value_len] = '\0';
                std::cout << " = '" << value_data.data() << "'";
                break;
            }
            case GGUFValueType::ARRAY: {
                GGUFValueType array_type;
                uint64_t array_len;
                if (!file.read(reinterpret_cast<char*>(&array_type), sizeof(array_type)) ||
                    !file.read(reinterpret_cast<char*>(&array_len), sizeof(array_len))) {
                    std::cout << " = [failed to read array info]";
                    break;
                }
                std::cout << " = [array of " << array_len << " elements]";
                
                // Manually skip array data based on type
                if (array_type == GGUFValueType::STRING) {
                    for (uint64_t j = 0; j < array_len; ++j) {
                        uint64_t str_len;
                        if (!file.read(reinterpret_cast<char*>(&str_len), sizeof(str_len))) {
                            std::cout << " [skip failed at element " << j << "]";
                            goto next_key;
                        }
                        if (!file.seekg(str_len, std::ios::cur)) {
                            std::cout << " [skip failed at element " << j << " data]";
                            goto next_key;
                        }
                    }
                } else {
                    // For non-string arrays, skip based on element size
                    size_t element_size = 0;
                    switch (array_type) {
                        case GGUFValueType::UINT8: case GGUFValueType::INT8: element_size = 1; break;
                        case GGUFValueType::UINT16: case GGUFValueType::INT16: element_size = 2; break;
                        case GGUFValueType::UINT32: case GGUFValueType::INT32: case GGUFValueType::FLOAT32: element_size = 4; break;
                        case GGUFValueType::UINT64: case GGUFValueType::INT64: case GGUFValueType::FLOAT64: element_size = 8; break;
                        case GGUFValueType::BOOL: element_size = 1; break;
                        default: std::cout << " [unknown array type]"; goto next_key;
                    }
                    if (!file.seekg(array_len * element_size, std::ios::cur)) {
                        std::cout << " [skip failed for " << array_len << " elements]";
                        goto next_key;
                    }
                }
                break;
            }
            default: {
                std::cout << " = [skipped]";
                if (!skip_value(file, type)) {
                    std::cout << " [skip failed]";
                    goto next_key;
                }
                break;
            }
        }
        next_key:
        std::cout << std::endl;
        
        // Check if file is still good
        if (!file.good()) {
            std::cerr << "File stream error after key " << i << ", stopping metadata parsing" << std::endl;
            return 1;
        }
    }
    
    // Now parse tensor information section
    std::cout << "\n=== TENSOR INFORMATION ===" << std::endl;
    
    for (uint64_t i = 0; i < header.tensor_count; ++i) {
        // Read tensor name
        uint64_t name_len;
        if (!file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len))) {
            std::cerr << "Failed to read tensor name length for tensor " << i << std::endl;
            break;
        }
        
        std::vector<char> name_data(name_len + 1);
        if (!file.read(name_data.data(), name_len)) {
            std::cerr << "Failed to read tensor name for tensor " << i << std::endl;
            break;
        }
        name_data[name_len] = '\0';
        
        // Read tensor dimensions
        uint32_t n_dims;
        if (!file.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims))) {
            std::cerr << "Failed to read tensor dimensions count for tensor " << i << std::endl;
            break;
        }
        
        std::vector<uint64_t> dims(n_dims);
        if (!file.read(reinterpret_cast<char*>(dims.data()), n_dims * sizeof(uint64_t))) {
            std::cerr << "Failed to read tensor dimensions for tensor " << i << std::endl;
            break;
        }
        
        // Read tensor type and offset
        uint32_t tensor_type;
        uint64_t offset;
        if (!file.read(reinterpret_cast<char*>(&tensor_type), sizeof(tensor_type)) ||
            !file.read(reinterpret_cast<char*>(&offset), sizeof(offset))) {
            std::cerr << "Failed to read tensor type/offset for tensor " << i << std::endl;
            break;
        }
        
        // Print tensor information
        std::cout << "Tensor[" << i << "]: '" << name_data.data() << "' ";
        std::cout << "dims=[";
        for (uint32_t d = 0; d < n_dims; ++d) {
            if (d > 0) std::cout << ", ";
            std::cout << dims[d];
        }
        std::cout << "] type=" << tensor_type << " offset=" << offset << std::endl;
    }
    
    return 0;
}