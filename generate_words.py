import random
import string

# Generate 300 unique random lowercase words of length 1 to 8
def generate_random_word():
    length = random.randint(1, 8)
    return ''.join(random.choices(string.ascii_lowercase, k=length))

random.seed(42)  # for reproducibility
size = 10
word_set = set()
while len(word_set) < size:
    word_set.add(generate_random_word())

# Sort the words (important for LOUDS add() function which expects sorted keys)
sorted_words = sorted(word_set)

# Join into newline-separated text
dict_data = '\n'.join(sorted_words)
#dict_data[:1000]  # preview first 1000 characters

dict_path = f"data/words_{size}.dict"
with open(dict_path, "w") as f:
    f.write(dict_data)