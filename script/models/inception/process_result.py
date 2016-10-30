'''
Coding Just for Fun
Created by burness on 16/9/7.
'''
class_id_dict = {
    'guinea pig': 0,
    'squirrel': 1,
    'sikadeer': 2,
    'fox': 3,
    'dog': 4,
    'wolf': 5,
    'cat': 6,
    'chipmunk': 7,
    'giraffe': 8,
    'reindeer': 9,
    'hyena': 10,
    'weasel': 11
}
f_write = open('processed_results.txt','w')
with open('results.txt', 'r') as fread:
    for line in fread.readlines():
        line_list = line.split("\t")
        img_name = line_list[0].split('.')[0]
        top1_id = class_id_dict[line_list[1]]
        top1_score = line_list[2]
        top2_id =class_id_dict[line_list[3]]
        top2_score = line_list[4]
        line_after = img_name+'\t'+str(top1_id)+'\t'+top1_score+'\t'+str(top2_id)+'\t'+top2_score
        f_write.write(line_after)


