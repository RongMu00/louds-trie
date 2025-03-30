ALL:
	# g++ -O2 -Wall -Wextra -march=native *.cpp -o louds-trie
	# g++ -O2 -Wall -Wextra -mbmi2 *.cpp -o louds-trie
	g++ -O2 -Wall -Wextra *.cpp -o louds-trie
	#-DUSE_PDEP_SELECT 
