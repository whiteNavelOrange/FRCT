import pickle

# import random
# a=random.random()
# print(a)
# b=random.random()
# print(b)
with open('/root/data/relative_data/train/bimanual_pick_laptop/all_variations/episodes/episode0/variation_descriptions.pkl', 'rb') as f:
    data = pickle.load(f)


print(data)
