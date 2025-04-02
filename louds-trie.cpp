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
  Trie* merge_trie(const Trie& t2);

  uint64_t n_keys() const {
    return n_keys_;
  }
  uint64_t n_nodes() const {
    return n_nodes_;
  }
  uint64_t size() const {
    return size_;
  }

  friend Trie* merge_louds_tries_level(const Trie& t1, const Trie& t2);

 private:
  vector<Level> levels_;
  uint64_t n_keys_;
  uint64_t n_nodes_;
  uint64_t size_;
  string last_key_;
};

// merge by level
Trie* merge_louds_tries_level(const Trie& t1, const Trie& t2) {
  const TrieImpl* impl1 = t1.get_impl();
  const TrieImpl* impl2 = t2.get_impl();

  TrieImpl* merged_impl = new TrieImpl();
  Trie* merged_trie = new Trie();
  delete merged_trie->get_impl();
  merged_trie->set_impl(merged_impl);

  // reset level 0 
  merged_impl->levels_.clear();
  merged_impl->levels_.resize(2);
  merged_impl->levels_[0].louds = BitVector();
  merged_impl->levels_[0].outs  = BitVector();
  merged_impl->levels_[0].labels.clear();
  merged_impl->levels_[0].louds.add(0);
  merged_impl->levels_[0].louds.add(1);
  merged_impl->levels_[0].outs.add(0);
  merged_impl->levels_[0].labels.push_back(' ');

  // determine maximum depth from both tries
  size_t max_depth = std::max(impl1->levels_.size(), impl2->levels_.size());
  if (merged_impl->levels_.size() < max_depth)
    merged_impl->levels_.resize(max_depth);

  // define a NodeMapping that carries (merged_id, level, and origins from input tries)
  struct NodeMapping {
      uint64_t merged_id;
      std::vector<std::pair<int, uint64_t>> origins; // (trie idx, original node id)
      uint64_t level;
  };

  // initialize BFS with the merged root - level 0
  std::vector<NodeMapping> current_level_nodes;
  current_level_nodes.push_back({0, {{1, 0}, {2, 0}}, 0});
  merged_impl->n_nodes_ = 1;
  merged_impl->n_keys_  = 0;
  uint64_t global_id = 1;  // next merged node id

  // Process levels 0 .. max_depth-2 - compute children for each node
  for (size_t level = 0; level < max_depth - 1; ++level) {
    // Ensure level (level+1) exists.
    if (merged_impl->levels_.size() <= level + 1)
      merged_impl->levels_.resize(level + 2);
    Level& merged_level = merged_impl->levels_[level + 1];

    // Reset merged level data.
    merged_level.louds = BitVector();
    merged_level.outs  = BitVector();
    merged_level.labels.clear();

    std::vector<NodeMapping> next_level_nodes;

    // Process each parent in current_level_nodes separately
    for (const auto& parent : current_level_nodes) {
      // For this parent, aggregate children from all its origins
      std::map<char, std::pair<bool, std::vector<std::pair<int, uint64_t>>>> childrenMap;
      for (const auto& origin : parent.origins) {
        int trie_idx = origin.first;
        uint64_t orig_id = origin.second;
        const TrieImpl* impl = (trie_idx == 1) ? impl1 : impl2;
        if (parent.level + 1 >= impl->levels_.size()) continue;
        const Level &input_level = impl->levels_[parent.level + 1];

        uint64_t node_pos = 0, base_node_id = 0;
        if (orig_id != 0) {
          node_pos = input_level.louds.select(orig_id - 1) + 1;
          base_node_id = node_pos - orig_id;
        } else {
          node_pos = 0;
          base_node_id = 0;
        }
        // Scan children for this origin until a termination (1-bit) is found
        for (uint64_t pos = node_pos; pos < input_level.louds.n_bits; ++pos) {
          if (input_level.louds.get(pos)) break;
          uint64_t child_id = pos - node_pos + base_node_id;
          if (child_id >= input_level.labels.size()) continue;
          char label = input_level.labels[child_id];
          bool is_terminal = input_level.outs.get(child_id);
          auto &entry = childrenMap[label];
          entry.first = entry.first || is_terminal;
          entry.second.push_back({trie_idx, child_id});
        }
      }
      // for this parent, output its children list
      // for a parent's children list, output one 0-bit per child then a terminating 1
      if (childrenMap.empty()) {
        // No children: output a termination bit.
        merged_level.louds.add(1);
      } else {
        // Convert the map to a sorted vector by label
        std::vector<std::tuple<char, bool, std::vector<std::pair<int, uint64_t>>>> sortedChildren;
        for (auto &p : childrenMap) {
          sortedChildren.push_back({p.first, p.second.first, p.second.second});
        }
        std::sort(sortedChildren.begin(), sortedChildren.end(),
                  [](auto &a, auto &b) { return std::get<0>(a) < std::get<0>(b); });
        // for each child, output a 0-bit and record its label and terminal flag
        for (size_t i = 0; i < sortedChildren.size(); ++i) {
          merged_level.louds.add(0);
          char label = std::get<0>(sortedChildren[i]);
          merged_level.labels.push_back(label);
          bool is_terminal = std::get<1>(sortedChildren[i]);
          merged_level.outs.add(is_terminal ? 1 : 0);
          if (is_terminal) merged_impl->n_keys_++;
          // create a new NodeMapping for this child
          NodeMapping childMapping;
          childMapping.merged_id = global_id++;
          childMapping.level = parent.level + 1;
          childMapping.origins = std::get<2>(sortedChildren[i]);
          next_level_nodes.push_back(childMapping);
        }
        // after all children for this parent, output a termination 1-bit
        merged_level.louds.add(1);
      }
    } // end for each parent

    current_level_nodes = next_level_nodes;
  } // end for each level

  // set last key
  std::set<std::string> all_keys;
  for (const auto& [key, _] : t1.enumerate_keys()) all_keys.insert(key);
  for (const auto& [key, _] : t2.enumerate_keys()) all_keys.insert(key);
  if (!all_keys.empty())
    merged_impl->last_key_ = *all_keys.rbegin();

  // trim trailing empty levels
  while (!merged_impl->levels_.empty()) {
    const auto& lvl = merged_impl->levels_.back();
    if (lvl.louds.n_bits <= 1 && lvl.labels.empty())
      merged_impl->levels_.pop_back();
    else break;
  }

  // reset offsets 
  for (size_t i = 0; i < merged_impl->levels_.size(); ++i)
    merged_impl->levels_[i].offset = 0;

  // build 
  merged_impl->build();

  return merged_trie;
}

// build a trie structure from all unique keys and convert to louds format and traverse by level 
// Trie* merge_louds_tries_level_v1(const Trie& t1, const Trie& t2) {
//   const TrieImpl* impl1 = t1.get_impl();
//   const TrieImpl* impl2 = t2.get_impl();

//   // Extract all expected keys for verification
//   std::set<std::string> expected_keys;
//   auto keys1 = t1.enumerate_keys();
//   auto keys2 = t2.enumerate_keys();
//   for (const auto& [key, _] : keys1) expected_keys.insert(key);
//   for (const auto& [key, _] : keys2) expected_keys.insert(key);
  
//   std::cout << "Total unique keys from both tries: " << expected_keys.size() << std::endl;

//   // create a new trie
//   TrieImpl* merged_impl = new TrieImpl();
//   Trie* merged_trie = new Trie();

//   delete merged_trie->get_impl();
//   merged_trie->set_impl(merged_impl);

//   // build in-memory trie from all keys 
//   struct TrieNode {
//     std::map<char, TrieNode*> children;
//     bool is_terminal;
    
//     TrieNode() : children(), is_terminal(false) {}
//     ~TrieNode() {
//       for (auto& [_, child] : children) {
//         delete child;
//       }
//     }
//   };

//   TrieNode* root = new TrieNode();
//   for (const auto& key : expected_keys) {
//     TrieNode* node = root;
//     for (char c : key) {
//       if (node->children.find(c) == node->children.end()) {
//         node->children[c] = new TrieNode();
//       }
//       node = node->children[c];
//     }
//     node->is_terminal = true;
//   }

//   // Initialize merged structure
//   merged_impl->levels_.clear();
//   merged_impl->levels_.resize(2);  // Start with levels 0 and 1
//   merged_impl->n_nodes_ = 1;  // Root node
//   merged_impl->n_keys_ = 0;
  
//   // initialize root level (level 0)
//   merged_impl->levels_[0].louds.add(0);
//   merged_impl->levels_[0].louds.add(1);
//   merged_impl->levels_[0].outs.add(0);
//   merged_impl->levels_[0].labels.push_back(' ');
  
//   // initialize level 1
//   merged_impl->levels_[1].louds.add(1);
  
//   // convert in-memory trie to LOUDS format with BFS traversal
//   std::vector<std::pair<TrieNode*, uint64_t>> current_level;
//   current_level.push_back({root, 0});  // (node, level)
  
//   for (uint64_t level = 0; !current_level.empty(); level++) {
//     std::vector<std::pair<TrieNode*, uint64_t>> next_level;
    
//     // ensure we have enough levels
//     if (level + 1 >= merged_impl->levels_.size()) {
//       merged_impl->levels_.resize(level + 2);
//     }
    
//     Level& current_level_struct = merged_impl->levels_[level + 1];
    
//     for (auto& [node, _] : current_level) {
//       std::vector<std::pair<char, TrieNode*>> sorted_children(node->children.begin(), node->children.end());
//       // add children to LOUDS structure
//       for (auto& [label, child] : sorted_children) {
//         // add 0-bit for this child
//         current_level_struct.louds.add(0);
//         // record terminal status
//         current_level_struct.outs.add(child->is_terminal ? 1 : 0);
//         // add label
//         current_level_struct.labels.push_back(label);
//         // count terminal nodes
//         if (child->is_terminal) {
//           merged_impl->n_keys_++;
//         }
//         // add to node count
//         merged_impl->n_nodes_++;
//         // queue for next level
//         next_level.push_back({child, level + 1});
//       }
      
//       // add 1-bit to mark end of children list
//       current_level_struct.louds.add(1);
//     }
    
//     std::cout << "Level " << (level + 1) << ": "
//               << "louds bits = " << current_level_struct.louds.n_bits
//               << ", outs = " << current_level_struct.outs.n_bits
//               << ", labels = " << current_level_struct.labels.size()
//               << std::endl;
    
//     current_level = next_level;
//   }
  
//   // clean up
//   delete root;
  
//   // set last key
//   if (!expected_keys.empty()) {
//     merged_impl->last_key_ = *expected_keys.rbegin();
//   }
  
//   // build
//   merged_impl->build();
  
//   return merged_trie;
// }

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

// Fixed enumerate_keys implementation
std::vector<std::pair<std::string, int64_t>> TrieImpl::enumerate_keys() const {
  std::vector<std::pair<std::string, int64_t>> results;
  
  // Early exit for empty tries
  if (n_keys_ == 0) {
    return results;
  }
  
  // Check if the root is terminal (empty string)
  if (levels_[0].outs.get(0)) {
    results.emplace_back("", levels_[0].offset);
  }
  
  // Stack for DFS traversal: (level, node_id, prefix)
  std::vector<std::tuple<uint64_t, uint64_t, std::string>> stack;
  stack.push_back(std::make_tuple(0, 0, ""));
  
  while (!stack.empty()) {
    uint64_t level, node_id;
    std::string prefix;
    std::tie(level, node_id, prefix) = stack.back();
    stack.pop_back();
    
    // Skip if we've reached max depth
    if (level + 1 >= levels_.size()) continue;
    
    const auto& next_level = levels_[level + 1];
    
    // Find position of children in the next level
    uint64_t node_pos, base_id;
    if (node_id != 0) {
      // Check if this node has any children (if node_id-1 is within range of 1-bits)
      if (node_id - 1 < next_level.louds.rank(next_level.louds.n_bits)) {
        node_pos = next_level.louds.select(node_id - 1) + 1;
        base_id = node_pos - node_id;
      } else {
        continue; // No children
      }
    } else {
      node_pos = 0;
      base_id = 0;
    }
    
    // Process all children of this node
    for (uint64_t pos = node_pos; pos < next_level.louds.n_bits; ++pos) {
      if (next_level.louds.get(pos)) {
        // End of children list
        break;
      }
      
      uint64_t child_id = pos - node_pos + base_id;
      
      if (child_id < next_level.labels.size()) {
        char label = next_level.labels[child_id];
        std::string new_prefix = prefix + label;
        
        // Check if terminal
        if (next_level.outs.get(child_id)) {
          int64_t key_id = next_level.offset + next_level.outs.rank(child_id);
          results.emplace_back(new_prefix, key_id);
        }
        
        // Continue traversal
        stack.push_back(std::make_tuple(level + 1, child_id, new_prefix));
      }
    }
  }
  
  // Sort results for consistent ordering
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

Trie* Trie::merge_trie_naive(const Trie& t2) {
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


// Update Trie::merge_trie to use this function
Trie* Trie::merge_trie(const Trie& t2) {
  return merge_louds_tries_level(*this, t2);
}

std::vector<std::pair<std::string, int64_t>> Trie::enumerate_keys() const {
  return impl_->enumerate_keys();
}


}  // namespace louds
