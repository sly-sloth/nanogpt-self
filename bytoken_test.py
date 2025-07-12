import os
import time
from pathlib import Path
from bytoken.bytoken import ByToken

dataset_dir = Path("dataset/harry-potter-books") 


train_corpus = ""

for book in os.listdir(dataset_dir)[:1]:
    book_path = dataset_dir / book
    with open(book_path, 'r') as f:
        train_corpus += f.read()

print(f"corpus size: {len(train_corpus)}")


enc = ByToken()
start_time = time.time()
enc.train(train_corpus, 256, verbose=True)
end_time = time.time()

print(f"completed in {(end_time-start_time):.3f}s")


sample = """But on the edge of town, drills were driven out of his mind by something else. As he sat in the usual morning traffic jam, he couldn’t help noticing that there seemed to be a lot of strangely dressed people about. People in cloaks. Mr. Dursley couldn’t bear people who dressed in funny clothes — the getups you saw on young people! He supposed this was some stupid new fashion. He drummed his fingers on the steering wheel and his eyes fell on a huddle of these weirdos standing quite close by. They were whispering excitedly together. Mr. Dursley was enraged to see that a couple of them weren’t young at all; why, that man had to be older than he was, and wearing an emerald-green cloak! The nerve of him! But then it struck Mr. Dursley that this was probably some silly stunt —these people were obviously collecting for something…yes, that would be it. The traffic moved on and a few minutes later, Mr. Dursley arrived in the Grunnings parking lot, his mind back on drills."""


start_time = time.time()

for i in range(50000):
    if i % 1000 == 0:
        print(f"iter: {i}")

    encoded = enc.encode(sample)
    # print("encoded:", encoded)

    decoded = enc.decode(encoded)
    # print("decoded:", decoded)

end_time = time.time()

print(f"Total time taken: {end_time - start_time:.4f} s")

