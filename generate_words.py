import random
import string

def generate_random_word():
    length = random.randint(1, 8)
    return ''.join(random.choices(string.ascii_lowercase, k=length))

random.seed(40)  
size = 300
word_set = set()
while len(word_set) < size:
    word_set.add(generate_random_word())

sorted_words = sorted(word_set)

dict_data = '\n'.join(sorted_words)
dict_path = f"data/words_{size}_2.dict"
with open(dict_path, "w") as f:
    f.write(dict_data)