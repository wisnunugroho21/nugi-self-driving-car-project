import os

imgs   = list(sorted(os.listdir(os.path.join('', 'dataset/images'))))
masks  = list(sorted(os.listdir(os.path.join('', 'dataset/annotations/trimaps'))))

new_imgs = []
for img in imgs:
    new_imgs.append(img.split('.jpg')[0])

new_masks = []
for mask in masks:
    new_masks.append(mask.split('.png')[0])

""" not_same = []
for img in imgs """

not_same_img = []
for img in new_imgs:
    if img not in new_masks:
        not_same_img.append(img)


# print(len(masks))

not_same_mask = []
for mask in new_masks:
    if mask not in new_imgs:
        not_same_mask.append(mask)

print(len(not_same_img))
print(len(not_same_mask))

print(not_same_img)
print(not_same_mask)
