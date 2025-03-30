#include "louds-trie.hpp"

#ifdef _MSC_VER
 #include <intrin.h>
 #include <immintrin.h>
#else  // _MSC_VER
 #include <x86intrin.h>
#endif  // _MSC_VER

#include <cassert>
#include <vector>

namespace louds {
namespace {

uint64_t Popcnt(uint64_t x) {
#ifdef _MSC_VER
  return __popcnt64(x);
#else  // _MSC_VER
  return __builtin_popcountll(x);
#endif  // _MSC_VER
}

uint64_t Ctz(uint64_t x) {
#ifdef _MSC_VER
  return _tzcnt_u64(x);
#else  // _MSC_VER
  return __builtin_ctzll(x);
#endif  // _MSC_VER
}

struct BitVector {
  struct Rank {
    uint32_t abs_hi; // absolute rank high
    uint8_t abs_lo; // absolute rank low
    uint8_t rels[3]; // relative ranks of idx=1,2,3 words in the block

    uint64_t abs() const { // Returns the total number of 1-bits before this rank block
      return ((uint64_t)abs_hi << 8) | abs_lo;
    }
    void set_abs(uint64_t abs) {
      abs_hi = (uint32_t)(abs >> 8);
      abs_lo = (uint8_t)abs;
    }
  };

  vector<uint64_t> words; // a vector of 64-bit words
  vector<Rank> ranks; 
  vector<uint32_t> selects;
  uint64_t n_bits;

  BitVector() : words(), ranks(), selects(), n_bits(0) {}

  uint64_t get(uint64_t i) const {
    // get the i-th bit (get the word -> get the bit -> mask (LSB))
    return (words[i / 64] >> (i % 64)) & 1UL;
  }

  void set(uint64_t i, uint64_t bit) {
    if (bit) { // set the i-th bit to 1
      words[i / 64] |= (1UL << (i % 64));
    } else { // clear the i-th bit (set the i-th bit to 0)
      words[i / 64] &= ~(1UL << (i % 64));
    }
  }

  void add(uint64_t bit) { // add a bit to the end of bit vector
    if (n_bits % 256 == 0) { // if you are writing to a new rank block
      words.resize((n_bits + 256) / 64); // resize words to the next 256 bits (next 4 words)
    }
    set(n_bits, bit); // set the last bit of the bit vector to bit
    ++n_bits; // n_bits is the number of bits that acutally write to 
  }
  
  // build builds indexes for rank and select.
  void build() {
    uint64_t n_blocks = words.size() / 4;  // only one word at initialization, n_blocks = 0
    uint64_t n_ones = 0; 
    ranks.resize(n_blocks + 1); // at least one rank block
    for (uint64_t block_id = 0; block_id < n_blocks; ++block_id) {
      ranks[block_id].set_abs(n_ones);
      for (uint64_t j = 0; j < 4; ++j) {
        if (j != 0) {
          uint64_t rel = n_ones - ranks[block_id].abs();
          ranks[block_id].rels[j - 1] = rel;
        }

        uint64_t word_id = (block_id * 4) + j;
        uint64_t word = words[word_id];
        uint64_t n_pops = Popcnt(word);
        uint64_t new_n_ones = n_ones + n_pops;
        if (((n_ones + 255) / 256) != ((new_n_ones + 255) / 256)) {
          uint64_t count = n_ones;
          while (word != 0) {
            uint64_t pos = Ctz(word);
            if (count % 256 == 0) {
              selects.push_back(((word_id * 64) + pos) / 256);
              break;
            }
            word ^= 1UL << pos;
            ++count;
          }
        }
        n_ones = new_n_ones;
      }
    }
    ranks.back().set_abs(n_ones);
    selects.push_back(words.size() * 64 / 256);
  }

  // rank returns the number of 1-bits in the range [0, i).
  uint64_t rank(uint64_t i) const {
    uint64_t word_id = i / 64;
    uint64_t bit_id = i % 64;
    uint64_t rank_id = word_id / 4;
    uint64_t rel_id = word_id % 4;
    uint64_t n = ranks[rank_id].abs();
    if (rel_id != 0) {
      n += ranks[rank_id].rels[rel_id - 1];
    }
    n += Popcnt(words[word_id] & ((1UL << bit_id) - 1));
    return n;
  }
  
  // select returns the position of the (i+1)-th 1-bit （idx=i).
  uint64_t select(uint64_t i) const {
    const uint64_t block_id = i / 256;
    uint64_t begin = selects[block_id];
    uint64_t end = selects[block_id + 1] + 1UL;
    if (begin + 10 >= end) {
      while (i >= ranks[begin + 1].abs()) {
        ++begin;
      }
    } else {
      while (begin + 1 < end) {
        const uint64_t middle = (begin + end) / 2;
        if (i < ranks[middle].abs()) {
          end = middle;
        } else {
          begin = middle;
        }
      }
    }
    const uint64_t rank_id = begin;
    i -= ranks[rank_id].abs();

    uint64_t word_id = rank_id * 4;
    if (i < ranks[rank_id].rels[1]) {
      if (i >= ranks[rank_id].rels[0]) {
        word_id += 1;
        i -= ranks[rank_id].rels[0];
      }
    } else if (i < ranks[rank_id].rels[2]) {
      word_id += 2;
      i -= ranks[rank_id].rels[1];
    } else {
      word_id += 3;
      i -= ranks[rank_id].rels[2];
    }
    
    #ifdef USE_PDEP_SELECT
    return (word_id * 64) + Ctz(_pdep_u64(1UL << i, words[word_id]));
    #else
    // Fallback software version：find position of the (i+1)-th 1-bit in the word
    uint64_t word = words[word_id];
    int count = 0;
    for (int bit = 0; bit < 64; ++bit) {
      if ((word >> bit) & 1) {
        if (count == (int)i) {
          return (word_id * 64) + bit;
        }
        ++count;
      }
    }
    return UINT64_MAX; 
    #endif
  }

  uint64_t size() const {
    return sizeof(uint64_t) * words.size()
      + sizeof(Rank) * ranks.size()
      + sizeof(uint32_t) * selects.size();
  }
};

struct Level {
  BitVector louds;
  BitVector outs;
  vector<uint8_t> labels;
  uint64_t offset;

  Level() : louds(), outs(), labels(), offset(0) {}

  uint64_t size() const;
};

uint64_t Level::size() const {
  return louds.size() + outs.size() + labels.size();
}

}  // namespace

class TrieImpl {
 public:
  TrieImpl();
  ~TrieImpl() {}

  void add(const string &key);
  void build();
  int64_t lookup(const string &query) const;
  std::vector<std::pair<std::string, int64_t>> enumerate_keys() const;

  uint64_t n_keys() const {
    return n_keys_;
  }
  uint64_t n_nodes() const {
    return n_nodes_;
  }
  uint64_t size() const {
    return size_;
  }

 private:
  vector<Level> levels_;
  uint64_t n_keys_;
  uint64_t n_nodes_;
  uint64_t size_;
  string last_key_;
};

TrieImpl::TrieImpl()
  : levels_(2), n_keys_(0), n_nodes_(1), size_(0), last_key_() {
  levels_[0].louds.add(0);
  levels_[0].louds.add(1);
  levels_[1].louds.add(1);
  levels_[0].outs.add(0);
  levels_[0].labels.push_back(' ');
}

void TrieImpl::add(const string &key) {
  // if (!(key > last_key_)) {
  //   std::cerr << "Skipped invalid key: " << key << " (last: " << last_key_ << ")\n";
  // }
  assert(key > last_key_);
  if (key.empty()) {
    levels_[0].outs.set(0, 1);
    ++levels_[1].offset;
    ++n_keys_;
    return;
  }
  if (key.length() + 1 >= levels_.size()) {
    levels_.resize(key.length() + 2);
  }
  uint64_t i = 0;
  for ( ; i < key.length(); ++i) {
    Level &level = levels_[i + 1];
    uint8_t byte = key[i];
    if ((i == last_key_.length()) || (byte != level.labels.back())) {
      level.louds.set(levels_[i + 1].louds.n_bits - 1, 0);
      level.louds.add(1);
      level.outs.add(0);
      level.labels.push_back(key[i]);
      ++n_nodes_;
      break;
    }
  }
  for (++i; i < key.length(); ++i) {
    Level &level = levels_[i + 1];
    level.louds.add(0);
    level.louds.add(1);
    level.outs.add(0);
    level.labels.push_back(key[i]);
    ++n_nodes_;
  }
  levels_[key.length() + 1].louds.add(1);
  ++levels_[key.length() + 1].offset;
  levels_[key.length()].outs.set(levels_[key.length()].outs.n_bits - 1, 1);
  ++n_keys_;
  last_key_ = key;
}

void TrieImpl::build() {
  uint64_t offset = 0;
  for (uint64_t i = 0; i < levels_.size(); ++i) {
    Level &level = levels_[i];
    level.louds.build();
    level.outs.build();
    offset += levels_[i].offset;
    level.offset = offset;
    size_ += level.size();
  }
}

int64_t TrieImpl::lookup(const string &query) const {
  if (query.length() >= levels_.size()) {
    return false;
  }
  uint64_t node_id = 0;
  for (uint64_t i = 0; i < query.length(); ++i) {
    const Level &level = levels_[i + 1];
    uint64_t node_pos;
    if (node_id != 0) {
      node_pos = level.louds.select(node_id - 1) + 1;
      node_id = node_pos - node_id;
    } else {
      node_pos = 0;
    }

    // Linear search implementation
    // for (uint8_t byte = query[i]; ; ++node_pos, ++node_id) {
    //   if (level.louds.get(node_pos) || level.labels[node_id] > byte) {
    //     return -1;
    //   }
    //   if (level.labels[node_id] == byte) {
    //     break;
    //   }
    // }

    // Binary search implementation
    uint64_t end = node_pos;
    uint64_t word = level.louds.words[end / 64] >> (end % 64);
    if (word == 0) {
      end += 64 - (end % 64);
      word = level.louds.words[end / 64];
      while (word == 0) {
        end += 64;
        word = level.louds.words[end / 64];
      }
    }
    end += Ctz(word);
    uint64_t begin = node_id;
    end = begin + end - node_pos;

    uint8_t byte = query[i];
    while (begin < end) {
      node_id = (begin + end) / 2;
      if (byte < level.labels[node_id]) {
        end = node_id;
      } else if (byte > level.labels[node_id]) {
        begin = node_id + 1;
      } else {
        break;
      }
    }
    if (begin >= end) {
      return -1;
    }
  }
  const Level &level = levels_[query.length()];
  if (!level.outs.get(node_id)) {
    return false;
  }
  return level.offset + level.outs.rank(node_id);
}

// std::vector<std::pair<std::string, int64_t>> TrieImpl::enumerate_keys() const {
//   std::vector<std::pair<std::string, int64_t>> results;

//   struct Frame {
//     int level;
//     uint64_t node_id;
//     std::string prefix;
//   };

//   cout << "TrieImpl::enumerate_keys() called\n";

//   std::vector<Frame> stack;
//   stack.push_back({0, 0, ""});  // start from root

//   while (!stack.empty()) {
//     Frame cur = stack.back();
//     stack.pop_back();

//     if (static_cast<size_t>(cur.level + 1) >= levels_.size()) continue;
//     const auto& level = levels_[cur.level + 1];

//     uint64_t node_pos;
//     if (cur.node_id != 0)
//       node_pos = level.louds.select(cur.node_id - 1) + 1;
//     else
//       node_pos = 0;

//     uint64_t end = node_pos;
//     uint64_t word = level.louds.words[end / 64] >> (end % 64);
//     if (word == 0) {
//       end += 64 - (end % 64);
//       word = level.louds.words[end / 64];
//       while (word == 0) {
//         end += 64;
//         word = level.louds.words[end / 64];
//       }
//     }
//     end += __builtin_ctzll(word);
//     uint64_t begin = cur.node_id;
//     end = begin + end - node_pos;

//     for (uint64_t child = begin; child < end; ++child) {
//       std::string next_prefix = cur.prefix + (char)level.labels[child];
//       if (level.outs.get(child)) {
//         int64_t key_id = level.offset + level.outs.rank(child);
//         results.emplace_back(next_prefix, key_id);
//       }
//       stack.push_back({cur.level + 1, child, next_prefix});
//     }
//   }

//   cout << "TrieImpl::enumerate_keys() finished\n";

//   std::sort(results.begin(), results.end());
//   return results;
// }

std::vector<std::pair<std::string, int64_t>> TrieImpl::enumerate_keys() const {
  std::vector<std::pair<std::string, int64_t>> results;
  
  // Use DFS to traverse the trie and collect all keys
  std::vector<std::tuple<uint64_t, uint64_t, std::string>> stack;
  stack.push_back({0, 0, ""}); // (level, node_id, prefix)
  
  while (!stack.empty()) {
    auto [level, node_id, prefix] = stack.back();
    stack.pop_back();
    
    // Check if this node marks the end of a key
    if (level > 0 && levels_[level].outs.get(node_id)) {
      int64_t key_id = levels_[level].offset + levels_[level].outs.rank(node_id);
      results.emplace_back(prefix, key_id);
    }
    
    // Skip if we've reached the max level
    if (level + 1 >= levels_.size()) continue;
    
    const Level& next_level = levels_[level + 1];
    uint64_t node_pos;
    if (node_id != 0) {
      node_pos = next_level.louds.select(node_id - 1) + 1;
      node_id = node_pos - node_id;
    } else {
      node_pos = 0;
    }
    
    // Find all children
    for (uint64_t pos = node_pos; pos < next_level.louds.n_bits; ++pos) {
      if (next_level.louds.get(pos)) break;
      
      uint64_t child_id = pos - node_pos + node_id;
      if (child_id < next_level.labels.size()) {
        char label = next_level.labels[child_id];
        stack.push_back({level + 1, child_id, prefix + label});
      }
    }
  }
  
  std::sort(results.begin(), results.end());
  return results;
}

Trie::Trie() : impl_(new TrieImpl) {}

Trie::~Trie() {
  delete impl_;
}

void Trie::add(const string &key) {
  return impl_->add(key);
}

void Trie::build() {
  impl_->build();
}

int64_t Trie::lookup(const string &query) const {
  return impl_->lookup(query);
}

uint64_t Trie::n_keys() const {
  return impl_->n_keys();
}

uint64_t Trie::n_nodes() const {
  return impl_->n_nodes();
}

uint64_t Trie::size() const {
  return impl_->size();
}

// Trie* Trie::merge_trie(const Trie& t2) {
//   auto keys1 = this->enumerate_keys();
//   auto keys2 = t2.enumerate_keys();

//   std::cout << "start merge_trie\n";
//   std::cout << "Trie 1: #keys = " << keys1.size() << ", #nodes = " << n_nodes() << ", size = " << size() << " bytes\n";
//   std::cout << "Trie 2: #keys = " << keys2.size() << ", #nodes = " << t2.n_nodes() << ", size = " << t2.size() << " bytes\n";

//   vector<string> merged_keys;
//   // for (const auto& key : keys1) {
//   //   merged_keys.push_back(key.first);
//   // }
//   // for (const auto& key : keys2) {
//   //   merged_keys.push_back(key.first);
//   // }
//   for (auto& [k, _] : keys1) merged_keys.push_back(k);
//   for (auto& [k, _] : keys2) merged_keys.push_back(k);
//   std::sort(merged_keys.begin(), merged_keys.end());
//   auto last = std::unique(merged_keys.begin(), merged_keys.end());
//   merged_keys.erase(last, merged_keys.end());

//   for (auto& key : merged_keys) {
//     std::cout << key << std::endl;
//   }
  
//   Trie* out = new Trie();
//   for (const auto& key : merged_keys)
//       out->add(key);
//   out->build();
//   return out;
// }

Trie* Trie::merge_trie(const Trie& t2) {
  // Extract all keys from both tries
  auto keys1 = this->enumerate_keys();
  auto keys2 = t2.enumerate_keys();
  
  // std::cout << "Merging tries...\n";
  // std::cout << "Trie 1: #keys = " << keys1.size() << ", #nodes = " << n_nodes() << ", size = " << size() << " bytes\n";
  // std::cout << "Trie 2: #keys = " << keys2.size() << ", #nodes = " << t2.n_nodes() << ", size = " << t2.size() << " bytes\n";
  
  // Collect unique keys
  std::vector<std::string> merged_keys;
  merged_keys.reserve(keys1.size() + keys2.size());
  
  for (const auto& [key, _] : keys1) {
    merged_keys.push_back(key);
  }
  for (const auto& [key, _] : keys2) {
    merged_keys.push_back(key);
  }
  
  // Sort and remove duplicates
  std::sort(merged_keys.begin(), merged_keys.end());
  auto last = std::unique(merged_keys.begin(), merged_keys.end());
  merged_keys.erase(last, merged_keys.end());
  
  // Create a new trie with the merged keys
  Trie* merged_trie = new Trie();
  for (const auto& key : merged_keys) {
    merged_trie->add(key);
  }
  merged_trie->build();
  
  return merged_trie;
}

std::vector<std::pair<std::string, int64_t>> Trie::enumerate_keys() const {
  return impl_->enumerate_keys();
}


}  // namespace louds
