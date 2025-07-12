#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>
#include "bytoken.h"

namespace fs = std::filesystem;

int main() {
    ByToken tokenizer;

    fs::path book_dir = "dataset/harry-potter-books/01 Harry Potter and the Sorcerers Stone train.txt";

    std::string train_corpus;

    // try {
    //     for (const auto& entry: fs::directory_iterator(dataset_dir)) {
    //         if (entry.is_regular_file()) {
    //             std::ifstream file(entry.path());
    //             if (!file) {
    //                 std::cerr << "Failed to open file: " << entry.path() << std::endl;
    //                 continue;
    //             }

    //             std::ostringstream buffer;
    //             buffer << file.rdbuf();
    //             train_corpus += buffer.str();
    //         }

    //         break;
    //     }

    //     std::cout << "corpus size: " << train_corpus.size() << std::endl;
    // } catch (const fs::filesystem_error& e) {
    //     std::cerr << "Filesystem error: " << e.what() << std::endl;
    // }

    std::ifstream file(book_dir);

    std::ostringstream buffer;
    buffer << file.rdbuf();
    train_corpus += buffer.str();

    std::cout << "corpus size: " << train_corpus.size() << std::endl;
    int vocab_size = 256;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    tokenizer.train(train_corpus, vocab_size, true);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Total time taken: " << duration.count() << " s" << std::endl;

    std::string sample = R"(But on the edge of town, drills were driven out of his mind by something else. As he sat in the usual morning traffic jam, he couldn’t help noticing that there seemed to be a lot of strangely dressed people about. People in cloaks. Mr. Dursley couldn’t bear people who dressed in funny clothes — the getups you saw on young people! He supposed this was some stupid new fashion. He drummed his fingers on the steering wheel and his eyes fell on a huddle of these weirdos standing quite close by. They were whispering excitedly together. Mr. Dursley was enraged to see that a couple of them weren’t young at all; why, that man had to be older than he was, and wearing an emerald-green cloak! The nerve of him! But then it struck Mr. Dursley that this was probably some silly stunt —these people were obviously collecting for something…yes, that would be it. The traffic moved on and a few minutes later, Mr. Dursley arrived in the Grunnings parking lot, his mind back on drills.)";

    auto start_time_2 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 50000; i++) {
        if (i % 1000 == 0) std::cout << "iter: " << i << "\n";
        std::vector<int> encoded = tokenizer.encode(sample);
        // std::cout << "encoded: ";
        // for (int id : encoded) std::cout << id << " ";
        // std::cout << std::endl;

        std::string decoded = tokenizer.decode(encoded);
        // std::cout << "decoded: " << decoded << std::endl;
    }

    auto end_time_2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration2 = end_time_2 - start_time_2;
    std::cout << "Total time taken: " << duration2.count() << " s" << std::endl;

}