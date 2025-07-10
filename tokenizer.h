#pragma once

#include <string>
#include <unordered_map>
#include <vector>


struct pair_hash {
    size_t operator()(const std::pair<int, int> &p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};


class BPETokenizer {
private:
    std::unordered_map<int, std::string> itos;
    std::unordered_map<std::string, int> stoi;
    int vocab_size;
    int max_key;
    std::unordered_map<std::pair<int, int>, int, pair_hash> merges;
    std::vector<std::pair<std::string, int>> final_vocab;

public:
    BPETokenizer();

    void train(std::string text_corpus, int vocab_size, bool verbose = false);
    std::vector<int> encode(std::string text);
    std::string decode(std::vector<int> idx);
};
