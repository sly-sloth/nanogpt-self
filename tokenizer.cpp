#include "tokenizer.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <set>

BPETokenizer::BPETokenizer() : vocab_size(0), max_key(0) {

}


/**
 * @brief Trains the tokenizer on the provided text corpus.
 * 
 * This function builds a vocabulary up to the given size by applying
 * Byte Pair Encoding (BPE) merges on the text.
 * 
 * @param text_corpus The input text corpus for training.
 * @param vocab_size Desired vocabulary size after training.
 * @param verbose Enable verbose output for debugging or progress tracking.
 */
void BPETokenizer::train(std::string text_corpus, int vocab_size, bool verbose) {
    this->vocab_size = vocab_size;
    std::set<char> text_corpus_unique(text_corpus.begin(), text_corpus.end());

    if (text_corpus_unique.size() > vocab_size) {
        std::ostringstream oss;
        oss << "vocab_size (" << vocab_size << ") must be greater than or equal to the number of unique chars (" << text_corpus_unique.size() << ") in the text corpus";
        throw std::runtime_error(oss.str());
    }

    for (char ch : text_corpus_unique) {
        std::string str(1, ch);
        stoi[str] = max_key;
        itos[max_key] = str;
        max_key++;
    }

    std::vector<int> text_idx;
    for (char ch : text_corpus) {
        std::string str(1, ch);
        text_idx.push_back(stoi[str]);
    }

    std::set<int> milestones{0};
    for (int i = 2; i < 11; i += 2) {
        float frac = 0.1 * i;
        int milestone = static_cast<int>(vocab_size * frac) - 1;
        milestones.insert(milestone);
    }

    while (max_key < vocab_size) {
        if (verbose) {
            if (milestones.find(max_key) != milestones.end()) {
                double progress = static_cast<double>(max_key + 1) / vocab_size;
                int percent_trained = static_cast<int>(std::round(progress * 100));
                std::cout << "Training progress " << std::to_string(percent_trained) << "%\n";
            }
        }

        std::unordered_map<std::pair<int, int>, int, pair_hash> pair_count;
        for (int i = 0; i < text_idx.size()-1; i++) {
            pair_count[std::make_pair(text_idx[i], text_idx[i+1])]++;
        }

        if (pair_count.empty()) break;

        auto max_pair = std::max_element(pair_count.begin(), pair_count.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            })->first;

        std::vector<int> new_text_idx;

        int i{0};
        while (i < text_idx.size()-1) {
            if (std::pair<int, int>{text_idx[i], text_idx[i+1]} == max_pair) {
                new_text_idx.push_back(max_key);
                i += 2;
            } else {
                new_text_idx.push_back(text_idx[i]);
                i++;
            }
        }

        std::string merged_pair = itos[max_pair.first] + itos[max_pair.second];
        itos[max_key] = merged_pair;
        stoi[merged_pair] = max_key;

        merges[max_pair] = max_key;
        max_key++;
        text_idx = std::move(new_text_idx);
    }

    final_vocab.clear();
    for (const auto& [str, id] : stoi) {
        final_vocab.emplace_back(str, id);
    }
    std::sort(final_vocab.begin(), final_vocab.end(),
        [](const auto& a, const auto& b) {
            return a.first.length() > b.first.length();
        });
    
    if (verbose) {
        std::cout << "Tokenizer successfully trained! Final vocab size: " << stoi.size() << "\n";
    }
}

/**
 * @brief Function to encode a given string.
 * 
 * @param text The text (string) which is to be encoded.
 * 
 * @return A vector of encoded integers.
*/
std::vector<int> BPETokenizer::encode(std::string text) {
    std::vector<int> encoded_idx;
    size_t pos = 0;

    while (pos < text.size()) {
        bool matched = false;

        for (const auto& [substr, subkey] : this->final_vocab) {
            size_t len = substr.length();

            if (pos + len <= text.length() && text.compare(pos, len, substr) == 0) {
                encoded_idx.push_back(subkey);
                pos += len;
                matched = true;
                break;
            }
        }

        if (!matched) {
            ++pos;
        }
    }
    
    return encoded_idx;
}

/**
 * @brief Function to decode a given vector of encoded indices.
 * 
 * @param idx The vector of encoded indices (encoding) which is to be decoded.
 * 
 * @return The decoded string.
*/
std::string BPETokenizer::decode(std::vector<int> idx)
{
    // add support for unsupported/unknown tokens
    std::string decoded_str;
    for (int i : idx) {
        decoded_str += itos[i];
    }

    return decoded_str;
}
