#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <fstream>

#include "louds-trie.hpp"

using namespace std;
using namespace std::chrono;

void test_merged_trie(const vector<string>&keys1, const vector<string>&keys2);

int main(int argc, char * argv[]) {
  if (argc == 2) {
    ios_base::sync_with_stdio(false);
    vector<string> keys;
    string line;
    ifstream f(argv[1]);
    while (getline(f, line)) {
      keys.push_back(line);
    }

    high_resolution_clock::time_point begin = high_resolution_clock::now();
    louds::Trie trie;
    for (uint64_t i = 0; i < keys.size(); ++i) {
      trie.add(keys[i]);
    }
    trie.build();
    high_resolution_clock::time_point end = high_resolution_clock::now();
    double elapsed = (double)duration_cast<nanoseconds>(end - begin).count();
    cout << "build = " << (elapsed / keys.size()) << " ns/key" << endl;

    cout << "#keys = " << trie.n_keys() << endl;
    cout << "#nodes = " << trie.n_nodes() << endl;
    cout << "size = " << trie.size() << " bytes" << endl;
    cout << "bytes/key = " << (double)trie.size() / trie.n_keys() << endl;

    vector<uint64_t> ids(keys.size());

    begin = high_resolution_clock::now();
    for (uint64_t i = 0; i < keys.size(); ++i) {
      ids[i] = trie.lookup(keys[i]);
    }
    end = high_resolution_clock::now();
    elapsed = (double)duration_cast<nanoseconds>(end - begin).count();
    cout << "seq. lookup = " << (elapsed / keys.size()) << " ns/key" << endl;

    sort(ids.begin(), ids.end());
    for (uint64_t i = 0; i < ids.size(); ++i) {
      assert(ids[i] == i);
    }

    random_shuffle(keys.begin(), keys.end());

    begin = high_resolution_clock::now();
    for (uint64_t i = 0; i < keys.size(); ++i) {
      assert(trie.lookup(keys[i]) != -1);
    }
    end = high_resolution_clock::now();
    elapsed = (double)duration_cast<nanoseconds>(end - begin).count();
    cout << "rnd. lookup = " << (elapsed / keys.size()) << " ns/key" << endl;
  }

  if(argc == 3) {
    ios_base::sync_with_stdio(false);
    vector<string> keys1;
    vector<string> keys2;
    string line;
    ifstream f1(argv[1]);
    ifstream f2(argv[2]);
    while (getline(f1, line)) {
      if (line == argv[1]) {
        break;
      }
      keys1.push_back(line);
    }
    while (getline(f2, line)) {
      keys2.push_back(line);
    }

    // printf("keys1.size() = %lu\n", keys1.size());
    // printf("keys2.size() = %lu\n", keys2.size());

    test_merged_trie(keys1, keys2);
  }

  return 0;
}

void test_merged_trie(const vector<string>&keys1, const vector<string>&keys2) {
  // Create two tries and add keys to them
  louds::Trie trie1, trie2;
  for(const auto& key : keys1) {
    trie1.add(key);
  }
  for(const auto& key : keys2) {
    trie2.add(key);
  }
  trie1.build();
  trie2.build();

  printf("Trie 1: #keys = %lu, #nodes = %lu, size = %lu bytes\n", trie1.n_keys(), trie1.n_nodes(), trie1.size());
  printf("Trie 2: #keys = %lu, #nodes = %lu, size = %lu bytes\n", trie2.n_keys(), trie2.n_nodes(), trie2.size());

  // Merge the tries
  louds::Trie* mergedTrie = trie1.merge_trie(trie2);

  printf("Merged Trie: #keys = %lu, #nodes = %lu, size = %lu bytes\n", mergedTrie->n_keys(), mergedTrie->n_nodes(), mergedTrie->size());

  vector<string> all_keys = keys1;
  all_keys.insert(all_keys.end(), keys2.begin(), keys2.end());
  sort(all_keys.begin(), all_keys.end());
  all_keys.erase(unique(all_keys.begin(), all_keys.end()), all_keys.end());

  set<string> expected(all_keys.begin(), all_keys.end());

  auto merged_keys = mergedTrie->enumerate_keys();
  cout << "Merged keys:\n";
  for (const auto& [key, _] : merged_keys) {
    cout << key << " ";
  }
  cout << endl;
  set<string> result_set;
  for (const auto& [key, _] : merged_keys) {
    result_set.insert(key);
  }

  cout << "Expected keys:\n";
  for (const auto& key : expected) {
    cout << key << " ";
  }
  cout << endl;
  cout << "Result keys:\n";
  for (const auto& key : result_set) {
    cout << key << " ";
  }
  cout << endl;

  assert(result_set == expected);
  cout << "Keys merged correctly.\n";

  for (const auto& key : expected) {
    assert(mergedTrie->lookup(key) != -1);
  }
  cout << "Lookup after merge works.\n";

  cout << "#keys = " << mergedTrie->n_keys() << ", #nodes = " << mergedTrie->n_nodes() << endl;
  cout << "Total size = " << mergedTrie->size() << " bytes" << endl;
  cout << "Bytes per key = " << (double)mergedTrie->size() / mergedTrie->n_keys() << endl;
}