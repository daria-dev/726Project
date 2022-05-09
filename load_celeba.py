import pandas as pd
import os
from shutil import copyfile

number_male = 1000
number_female = 1000
numer_old = 1000
number_young = 1000

root_dir = "img_align_celeba"
dest_dir = "celeba_selected"

attr = pd.read_csv("list_attr_celeba.txt", sep='\s+')

# get male examples
male_exmp = attr[attr["Male"] == 1].head(number_male)
male_exmp_file = male_exmp.index

for file in male_exmp_file:
    file_name = os.path.join(root_dir, file)
    copyfile(file_name, os.path.join(dest_dir, file))


curr_old = len(male_exmp[male_exmp["Young"] == -1])
curr_young = len(male_exmp[male_exmp["Young"] == 1])

# get female examples
female_exmp = attr[attr["Male"] == -1].head(number_female)
female_exmp_file = female_exmp.index

for file in female_exmp_file:
    file_name = os.path.join(root_dir, file)
    copyfile(file_name, os.path.join(dest_dir, file))

curr_old += len(female_exmp[female_exmp["Young"] == -1])
curr_young += len(female_exmp[female_exmp["Young"] == 1])

# get remaining old/young
selected = pd.concat([male_exmp, female_exmp])
available = attr.drop(selected.index)

if numer_old > curr_old:
    old_exmp = available[available["Young"] == -1].head(numer_old- curr_old)
    old_exmp_file = old_exmp.index

    for file in old_exmp_file:
        file_name = os.path.join(root_dir, file)
        copyfile(file_name, os.path.join(dest_dir, file))

if number_young > curr_young:
    young_exmp = available[available["Young"] == 1].head(number_young- curr_young)
    young_exmp_file = young_exmp.index

    for file in young_exmp_file:
        file_name = os.path.join(root_dir, file)
        copyfile(file_name, os.path.join(dest_dir, file))

