#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 21:34:19 2021

@author: zfd297
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sp_tsne_hand1 as sp_tsne
import glob
import os
from scipy import sparse
from itertools import product
from sklearn.cluster import KMeans

right_cell_order = pd.read_csv("right_oder_cellid.csv")["x"]
right_cell_order = np.array(right_cell_order)


def get_doublelet(file_name):
    
    name_part1 = "./doublelet_file/SVMlabel_" 
    name_part2 = "_lr0.5.txt"
    path = name_part1 + file_name + name_part2
    doublelet_list = pd.read_csv(path, sep= "\t",index_col=0)
    dic = {}
    
    for i in doublelet_list.index:
    
        double_cell = np.array(doublelet_list.loc[i])[0]
        double_cell = double_cell.replace("+", "")
        double_cell = double_cell.replace(file_name, "")
        if double_cell != "":
            tmp = np.where(right_cell_order == i)[0][0]
            if double_cell not in dic.keys():
                dic[double_cell] = []
                dic[double_cell].append(tmp)
            else:
                dic[double_cell].append(tmp)
                
    return dic


def point_distance_line(point,line_point1,line_point2):

    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(line_point1-line_point2)
    return distance


def find_center_point(re_sp, correct_double_group):
    
    
    
    double_coordinate = re_sp[np.array(correct_double_group[0]),:]
    kmeans = KMeans(n_clusters=1, random_state=0).fit(double_coordinate)
    center_point = kmeans.cluster_centers_
    #center_point = [np.median(double_coordinate[:,0]), np.median(double_coordinate[:,1])]
    
        
        
    return center_point[0]

def find_line_(center_point):
    
    all_lines = []
    for i in np.linspace(0,180,90):
        rad = math.radians(i)
        sin_len = np.sin(rad)*3
        cos_len = np.cos(rad)*3
        
        tmp = np.array([center_point[0]+cos_len, center_point[1]+sin_len])
        
        all_lines.append(tmp)
        
    return all_lines
        
def get_double_border(re_sp, group1, group2, center_point,line_point):
    
    sum_dis_list = []
    for k in line_point:
        
        dis_1 = point_distance_line(group1, center_point, k)
        dis_2 = point_distance_line(group2, center_point, k)
    
        sum_dis1 = np.sum(dis_1)
        sum_dis2 = np.sum(dis_2)
        
        final_sum_dis = sum_dis1 + sum_dis2
        
        sum_dis_list.append(final_sum_dis)
        
    sort_index = np.argsort(-np.array(sum_dis_list))
    
    final_line_point = line_point[sort_index[0]]
    
    return final_line_point
   


plt.figure(None,(20,15))

for i in range(len(unique_cell_id)):
    
    tmp = np.where(new_cell_id == unique_cell_id[i])[0]
    plt.scatter(new_re_sp[tmp,0], new_re_sp[tmp,1],  s=50, label=unique_cell_id[i],c = colors[i]  )

plt.scatter(re_sp[doublelet_half_1,0], re_sp[doublelet_half_1,1], s=50, c="black")
plt.scatter(re_sp[doublelet_half_2,0], re_sp[doublelet_half_2,1], s=15, c="grey")

plt.scatter(center_point_1[0], center_point_1[1], s=80,c="red")
plt.scatter(a1[0], a1[1], s=80,c="red")

        
doule_list = []
for filename in os.listdir("./doublelet_file/"):
    filename = filename.replace("SVMlabel_", "")
    filename = filename.replace("_lr0.5.txt", "")
    doule_list.append(filename)
    

double_info_each_group = {}
for i in doule_list:
    double_info_each_group[i] = get_doublelet(i)
    

final_double_dic = {}
for i in double_info_each_group.keys():
    tmp_dic_1 = double_info_each_group[i]
    for j in tmp_dic_1.keys():
        tmp_dic_2 = double_info_each_group[j]
        if i in tmp_dic_2.keys():
            combination_name = i+"+"+j
            combination_value_1 = tmp_dic_1[j]
            combination_value_2 = tmp_dic_2[i]
            final_double_dic[combination_name] = []
            final_double_dic[combination_name].append(combination_value_1)
            final_double_dic[combination_name].append(combination_value_2)
            

from scipy.io import mmread
from sklearn.decomposition import PCA

count_mat = mmread('normalized_data.mtx')
count_mat = count_mat.A
count_mat = count_mat.T
pca = PCA(n_components=30)
pca_mat = pca.fit_transform(count_mat)


P = sp_tsne.calculate_high_P(pca_mat,perplexity=50)
re = sp_tsne.run_tsne(pca_mat, P)


#############3
cell_id = pd.read_csv("right_oder_cellid.csv")
cell_id = cell_id['0']

unique_cell_id = np.unique(cell_id)

cell_id_num = cell_id.copy()

dict_id_num = dict(zip(cell_id, cell_id_num))

for i in range(len(unique_cell_id)):
    tmp = unique_cell_id[i]
    index = np.where(cell_id == tmp)[0]
    cell_id_num[index] = i
    

umaps = pd.read_csv("umap.txt",sep=" ")

colors = ["#FF8C00", "#32CD32", "#191970", "#00FA9A", "#7B68EE", "#FFD700", "#F08080", 
          "#FF0000", "#C71585", "#ffff14", "#00FFFF", "#680018", "#caa0ff"]

color_map = np.array(cell_id.copy())

for i in range(len(unique_cell_id)):
    index = np.where(cell_id == unique_cell_id[i])[0]
    color_map[index] = colors[i]
#############


plt.figure(None,(20,15))

for i in range(len(unique_cell_id)):
    
    tmp = np.where(cell_id == unique_cell_id[i])[0]
    plt.scatter(re[tmp,0], re[tmp,1],  s=50, label=unique_cell_id[i],c = colors[i]  )
plt.legend(markerscale=2, ncol= 4,prop={'size': 14})
plt.axis("off")


ss = arrange_border(re, correct_double_dic)


P_sp = P.copy()
P_sp = P_sp.A

all_doubel_list =[]
g1_list =[]
g2_list = []

for key in final_double_dic.keys():
    
    g1 = final_double_dic[key][0]
    g2 = final_double_dic[key][1]
   
    if len(g1) <= len(g2):
        
        g2 = g2[:len(g1)]
        
        all_doubel_list += g1
        all_doubel_list += g2
        
        g1_list += g1
        g2_list += g2
        
    else:
        g1 = g1[:len(g2)]
        
        all_doubel_list += g1
        all_doubel_list += g2
        
        g1_list += g1
        g2_list += g2
         
        
    g1 = np.array(g1)
    g2 = np.array(g2)    
    P_sp[g1,g2] = 3.1806615776081425e-03

g1_list = np.array(g1_list)
g2_list = np.array(g2_list)


########
P_sp = P.copy()
doublelet_cells = []

doublelet_half_1 = []
doublelet_half_2 = []

correct_double_dic = {}
for i in ['ExVE+EmVE', 'ExVE+FG', 'EmVE+ExVE', 'EmVE+DE', 'NC+NP', 'NC+FG', 'NC+MG', 'NP+NC', 'NP+DE']:
    
    combination  = final_double_dic[i]
    len1 = len(combination[0])
    len2 = len(combination[1])
    
    if len1 < len2:
        
        tmp1 = list(combination[0])
        tmp2 = list(combination[1][:len1])
        # P_sp[tmp1, tmp2] = 3.1806615776081425e-04
        # P_sp[tmp2, tmp1] = 3.1806615776081425e-04
        tmp = tmp1 + tmp2
        l1 = np.array([k[0] for k in list(product(tmp, tmp))])
        l2 = np.array([k[1] for k in list(product(tmp, tmp))])
        P_sp[l1,l2] = 1.1806615776081425e-04
        
        
        doublelet_cells+=list(tmp1)
        doublelet_cells+=list(tmp2)
        doublelet_half_1+=list(tmp1)
        doublelet_half_2+=list(tmp2)
        correct_double_dic[i] = [tmp1,tmp2]
    else:
        tmp1 = list(combination[0][:len2])
        tmp2 = list(combination[1])
        # P_sp[tmp1, tmp2] = 3.1806615776081425e-04
        # P_sp[tmp2, tmp1] = 3.1806615776081425e-04
        
        tmp = tmp1 + tmp2
        l1 = np.array([k[0] for k in list(product(tmp, tmp))])
        l2 = np.array([k[1] for k in list(product(tmp, tmp))])
        P_sp[l1,l2] = 1.1806615776081425e-04
                
        doublelet_cells+=list(tmp1)
        doublelet_cells+=list(tmp2)
        doublelet_half_1+=list(tmp1)
        doublelet_half_2+=list(tmp2)
        correct_double_dic[i] = [tmp1,tmp2]

P_sp = sparse.csr_matrix(P_sp)
re_sp  = sp_tsne.run_tsne(pca_mat, P_sp, n_components=2)


        
new_index = [True for i in range(len(pca_mat))]
new_index = np.array(new_index)
new_index[np.array(all_doubel_list)] = False


cell_id = np.array(cell_id)
new_cell_id = cell_id[new_index]
new_re_sp = re_sp[new_index,:]

plt.figure(None,(20,15))

for i in range(len(unique_cell_id)):
    
    tmp = np.where(new_cell_id == unique_cell_id[i])[0]
    plt.scatter(new_re_sp[tmp,0], new_re_sp[tmp,1],  s=50, label=unique_cell_id[i],c = colors[i]  )

plt.scatter(re_sp[doublelet_half_1,0], re_sp[doublelet_half_1,1], s=50, c="black")
plt.scatter(re_sp[doublelet_half_2,0], re_sp[doublelet_half_2,1], s=15, c="grey")

for k in ss:
    
    plt.scatter(k[0],k[1],s=200,c = "red")
plt.legend(markerscale=2, ncol= 4,prop={'size': 14})

plt.axis("off")
ss = arrange_border(re_sp, correct_double_dic)
#################
P_sp = P.copy()
doublelet_cells = []

doublelet_half_1 = []
doublelet_half_2 = []

correct_double_dic = {}
for i in ['NP+DE']:
    
    combination  = final_double_dic[i]
    len1 = len(combination[0])
    len2 = len(combination[1])
    
    if len1 < len2:
        
        tmp1 = list(combination[0])
        tmp2 = list(combination[1][:len1])
        # P_sp[tmp1, tmp2] = 3.1806615776081425e-04
        # P_sp[tmp2, tmp1] = 3.1806615776081425e-04
        tmp = tmp1 + tmp2
        l1 = np.array([k[0] for k in list(product(tmp, tmp))])
        l2 = np.array([k[1] for k in list(product(tmp, tmp))])
        P_sp[l1,l2] = 1.1806615776081425e-04
        
        
        doublelet_cells+=list(tmp1)
        doublelet_cells+=list(tmp2)
        doublelet_half_1+=list(tmp1)
        doublelet_half_2+=list(tmp2)
        correct_double_dic[i] = [tmp1,tmp2]
    else:
        tmp1 = list(combination[0][:len2])
        tmp2 = list(combination[1])
        # P_sp[tmp1, tmp2] = 3.1806615776081425e-04
        # P_sp[tmp2, tmp1] = 3.1806615776081425e-04
        
        tmp = tmp1 + tmp2
        l1 = np.array([k[0] for k in list(product(tmp, tmp))])
        l2 = np.array([k[1] for k in list(product(tmp, tmp))])
        P_sp[l1,l2] = 1.1806615776081425e-04
                
        doublelet_cells+=list(tmp1)
        doublelet_cells+=list(tmp2)
        doublelet_half_1+=list(tmp1)
        doublelet_half_2+=list(tmp2)
        correct_double_dic[i] = [tmp1,tmp2]

P_sp = sparse.csr_matrix(P_sp)
re_sp  = sp_tsne.run_tsne(pca_mat, P_sp, n_components=2)


        
new_index = [True for i in range(len(pca_mat))]
new_index = np.array(new_index)
new_index[np.array(all_doubel_list)] = False


cell_id = np.array(cell_id)
new_cell_id = cell_id[new_index]
new_re_sp = re_sp[new_index,:]



group1 = new_re_sp[np.where(new_cell_id=="NP")[0],:]
group2 = new_re_sp[np.where(new_cell_id=="DE")[0],:]
center_point_1 =  find_center_point(re_sp, correct_double_dic["NP+DE"])
line_point = find_line_(center_point_1)
border_line = get_double_border(re_sp, group1, group2,
                  center_point_1,line_point)

from math import sqrt, pow, acos
def angle_of_vector(v1, v2):
    pi = 3.1415
    vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
    length_prod = sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * sqrt(pow(v2[0], 2) + pow(v2[1], 2))
    cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
    return (acos(cos) / pi) * 180



slope = (border_line[1]-center_point_1[1]) / (border_line[0]-center_point_1[0])
slope_arc = math.atan(slope)
new_coord_list = []
for k in re_sp[doublelet_half_1,:]:

    v1 = k -  center_point_1
    v2 = center_point_1 - border_line
    angle = math.radians(angle_of_vector(v1,v2))
    lenth_1 = np.sqrt((k[0]-center_point_1[0])**2 + (k[1]-center_point_1[1])**2)
    lenth_2 = lenth_1 * math.cos(angle)
    
    projected_x = center_point_1[0] + math.cos(slope_arc)*lenth_2
    projected_y = center_point_1[1] + math.sin(slope_arc)*lenth_2

    tmp_projected_point = [projected_x,projected_y]
    new_coord_list.append(tmp_projected_point)
    
plt.figure(None,(20,15))

for i in range(len(unique_cell_id)):
    
    tmp = np.where(new_cell_id == unique_cell_id[i])[0]
    plt.scatter(new_re_sp[tmp,0], new_re_sp[tmp,1],  s=50, label=unique_cell_id[i],c = colors[i]  )


plt.scatter(aa[:,0], aa[:,1], s=50,c='black')
plt.scatter(re_sp[doublelet_half_1,0], re_sp[doublelet_half_1,1], s=50, c="black")
plt.scatter(re_sp[doublelet_half_2,0], re_sp[doublelet_half_2,1], s=15, c="grey")


    
plt.scatter(a1[0],a1[1],s=200,c = "red")
plt.scatter(center_point_1[0],center_point_1[1],s=200,c = "red")
plt.legend(markerscale=2, ncol= 4,prop={'size': 14})

plt.axis("off")
ss = arrange_border(re_sp, correct_double_dic)




#################


#######
# DE_NP = final_double_dic['DE+NP']
# P_sp = P.copy()
# for i in range(0,111):
#     tmp1 = DE_NP[0][i]
#     tmp2 = DE_NP[1][i]
#     P_sp[tmp1, tmp2] = 3.1806615776081425e-04
#     P_sp[tmp2, tmp1] = 3.1806615776081425e-04
#     P_sp[tmp1, tmp1] = 9.1806615776081425e-03
#     P_sp[tmp2, tmp2] = 9.1806615776081425e-03
    
# P_sp = sparse.csr_matrix(P_sp)
# re_sp  = sp_tsne.run_tsne(pca_mat, P_sp, n_components=3)

# all_double_list = DE_NP[0][:111] + DE_NP[1][:111]

# new_index = [True for i in range(len(pca_mat))]
# new_index = np.array(new_index)
# new_index[np.array(all_doubel_list)] = False

# cell_id = np.array(cell_id)
# new_cell_id = cell_id[new_index]
# new_re_sp = re_sp[new_index,:]

# g1_list = np.array(DE_NP[0][:111])
# g2_list = np.array(DE_NP[1][:111])

# plt.figure(None,(20,15))

# for i in range(len(unique_cell_id)):
    
#     tmp = np.where(new_cell_id == unique_cell_id[i])[0]
#     plt.scatter(new_re_sp[tmp,0], new_re_sp[tmp,1],  s=50, label=unique_cell_id[i],c = colors[i]  )

# plt.scatter(re_sp[g1_list,0], re_sp[g1_list,1], s=50, c="black")
# plt.scatter(re_sp[g2_list,0], re_sp[g2_list,1], s=50, c="grey")

# plt.legend(markerscale=2, ncol= 4,prop={'size': 14})

# plt.axis("off")


# #######
# P_sp = sparse.csr_matrix(P_sp)
# re_sp  = sp_tsne.run_tsne(pca_mat, P_sp, n_components=2)

# new_index = [True for i in range(len(pca_mat))]
# new_index = np.array(new_index)
# new_index[np.array(all_doubel_list)] = False

# cell_id = np.array(cell_id)
# new_cell_id = cell_id[new_index]
# new_re_sp = re_sp[new_index,:]


# plt.figure(None,(20,15))

# for i in range(len(unique_cell_id)):
    
#     tmp = np.where(new_cell_id == unique_cell_id[i])[0]
#     plt.scatter(new_re_sp[tmp,0], new_re_sp[tmp,1],  s=50, label=unique_cell_id[i],c = colors[i]  )

# plt.scatter(re_sp[g1_list,0], re_sp[g1_list,1], s=50, c="black")
# plt.scatter(re_sp[g2_list,0], re_sp[g2_list,1], s=50, c="grey")

# plt.legend(markerscale=2, ncol= 4,prop={'size': 14})

# plt.axis("off")

# ###plot 3d

# ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程


# for i in range(len(unique_cell_id)):
    
#     tmp = np.where(new_cell_id == unique_cell_id[i])[0]
#     ax.scatter(new_re_sp[tmp,0], new_re_sp[tmp,1],  new_re_sp[tmp,2], label=unique_cell_id[i],c = colors[i] )

# ax.scatter(re_sp[g1_list,0], re_sp[g1_list,1], re_sp[g1_list,2], c="grey")
# ax.scatter(re_sp[g2_list,0], re_sp[g2_list,1], re_sp[g1_list,2], c="grey")

# ax.legend(markerscale=2, ncol= 4,prop={'size': 14})

# ax.axis("off")





# re = sp_tsne.run_tsne(pca_mat, P, n_components=3)

# ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程


# for i in range(len(unique_cell_id)):
    
#     tmp = np.where(cell_id == unique_cell_id[i])[0]
#     ax.scatter(re[tmp,0], re[tmp,1], re[tmp,2] ,s=50, label=unique_cell_id[i],c = colors[i]  )


# ax.legend(markerscale=2, ncol= 4,prop={'size': 14})

# ax.axis("off")
##################
#################
##################

#################
P_sp = P.copy()
doublelet_cells = []

doublelet_half_1 = []
doublelet_half_2 = []

correct_double_dic = {}
cells_without_doublelet = {}
for i in ['DE+NP']:
    
    combination  = final_double_dic[i]
    len1 = len(combination[0])
    len2 = len(combination[1])
    
    if len1 < len2:
        
        tmp1 = list(combination[0])
        tmp2 = list(combination[1][:len1])
        # P_sp[tmp1, tmp2] = 3.1806615776081425e-04
        # P_sp[tmp2, tmp1] = 3.1806615776081425e-04
        tmp = tmp1 + tmp2
        l1 = np.array([k[0] for k in list(product(tmp, tmp))])
        l2 = np.array([k[1] for k in list(product(tmp, tmp))])
        P_sp[l1,l2] = 1.1806615776081425e-04
        
        
        doublelet_cells+=list(tmp1)
        doublelet_cells+=list(tmp2)
        doublelet_half_1+=list(tmp1)
        doublelet_half_2+=list(tmp2)
        correct_double_dic[i] = [tmp1,tmp2]
        
        group_name1 = i.split("+")[0]
        group_name2 = i.split("+")[1]
        
        
        
        
    
        
        cells_without_doublelet[i] = [tmp_left_i, tmp_right_i]
        
    else:
        tmp1 = list(combination[0][:len2])
        tmp2 = list(combination[1])
        # P_sp[tmp1, tmp2] = 3.1806615776081425e-04
        # P_sp[tmp2, tmp1] = 3.1806615776081425e-04
        
        tmp = tmp1 + tmp2
        l1 = np.array([k[0] for k in list(product(tmp, tmp))])
        l2 = np.array([k[1] for k in list(product(tmp, tmp))])
        P_sp[l1,l2] = 1.1806615776081425e-04
                
        doublelet_cells+=list(tmp1)
        doublelet_cells+=list(tmp2)
        doublelet_half_1+=list(tmp1)
        doublelet_half_2+=list(tmp2)
        correct_double_dic[i] = [tmp1,tmp2]
        
        
        tmp_left_i = combination[0]
        tmp_right_i = combination[1]
        
        for k in tmp1:
            tmp_left_i.remove(k)
            
        for k in tmp2:
            tmp_right_i.remove(k)
        
    
        
        cells_without_doublelet[i] = [tmp_left_i, tmp_right_i]

P_sp = sparse.csr_matrix(P_sp)
re_sp  = sp_tsne.run_tsne(pca_mat, P_sp, n_components=2)

