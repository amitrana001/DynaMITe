import torch
import os
# sam_path='datasets/SA1B/sa_000020'
sam_path = '/p/scratch/objectsegvideo/SA1B'
# sam_path ='/work/rana/sam_data/SA1B'
# save_path='datasets/SA1B/sa_000020/image_list.da'
# if not os.path.exists(save_path):
#     print("here")
#     os.mkdir(save_path)

data_folders = [name for name in os.listdir(sam_path) if (os.path.isdir(os.path.join(sam_path, name)) and 'sa_' in name)]
total_data_size = 0
for folder_name in data_folders:
    # print(f'Creating annotations for {folder_name}')
    folder_path = os.path.join(sam_path, folder_name)
    # save_path = os.path.join(folder_path, "image_list.da")
    # f_save = open(save_path, 'wb')
    # a=[]
    files = os.listdir(os.path.join(folder_path, 'images'))
    total_data_size += len(files)
    # for f in files:
    #     if f.split('.')[-1]=='jpg':
    #         a.append({'img_name': os.path.join(folder_path, 'images', f), 'ann_name': os.path.join(folder_path, 'annotations', f.split('.')[0]+'.json')})
    # torch.save(a, f_save)
    # f_save.close()
print(total_data_size)