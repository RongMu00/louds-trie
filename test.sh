#!/bin/bash

set -e

echo "🔧 Building LOUDS trie from example.dict..."
./louds-trie build example.dict

echo ""
echo "Running search queries:"
for word in an ant bar bat bee car cat dog elk; do
  echo -n "Search \"$word\": "
  ./louds-trie lookup example.dict "$word"
done