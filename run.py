#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 07:34:51 2021

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
from scipy.io import mmread
from sklearn.decomposition import PCA
from math import sqrt, pow, acos
import math
import scipy.optimize
import sympy
import random



def angle_of_vector(v1, v2):
    pi = 3.1415
    vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
    length_prod = sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * sqrt(pow(v2[0], 2) + pow(v2[1], 2))
    cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
    return (acos(cos) / pi) * 180



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
    import math
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


def run_sp_tsne(P,pca_mat,probability,group,cell_id):

    P_sp = P.copy()
    doublelet_cells = []
    doublelet_half_1 = []
    doublelet_half_2 = []
    correct_double_dic = {}
    for i in group:
        
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
            P_sp[l1,l2] = probability
            
            
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
            P_sp[l1,l2] = probability
                    
            doublelet_cells+=list(tmp1)
            doublelet_cells+=list(tmp2)
            doublelet_half_1+=list(tmp1)
            doublelet_half_2+=list(tmp2)
            correct_double_dic[i] = [tmp1,tmp2]
    
    P_sp = sparse.csr_matrix(P_sp)
    re_sp  = sp_tsne.run_tsne(pca_mat, P_sp, n_components=2)
    
    
            
    new_index = [True for i in range(len(pca_mat))]
    new_index = np.array(new_index)
    new_index[np.array(doublelet_cells)] = False
    
    
    cell_id = np.array(cell_id)
    new_cell_id = cell_id[new_index]
    new_re_sp = re_sp[new_index,:]
    
    
    return re_sp, new_re_sp, new_cell_id, correct_double_dic,doublelet_half_1


def get_final_double(group, new_cell_id, re_sp,correct_double_dic,doublelet_half_1):
    
    double_border_coord = {}
    for i in group:
        tmp_group1_index = i.split("+")[0]
        tmp_group2_index = i.split("+")[1]

        group1 = new_re_sp[np.where(new_cell_id==tmp_group1_index)[0],:]
        group2 = new_re_sp[np.where(new_cell_id==tmp_group2_index)[0],:]
        center_point_1 =  find_center_point(re_sp, correct_double_dic[i])
        line_point = find_line_(center_point_1)
        border_line = get_double_border(re_sp, group1, group2,
                          center_point_1,line_point)
        
        
        
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
            
        double_border_coord[i] = np.array(new_coord_list)
        
        vet_slope = -1/slope
        vet_slope_arc = math.atan(vet_slope)
        new_x = center_point_1 + math.cos(vet_slope_arc)*0.05
        new_y = center_point_1 + math.sin(vet_slope_arc)*0.05
        
       
        
        final_double_1 = []
        final_double_2 = []
        
        for j in new_coord_list:
            tmp_x = j[0] + math.cos(vet_slope_arc)*0.8
            tmp_y = j[1] + math.sin(vet_slope_arc)*0.8
            tmp_coord = [tmp_x, tmp_y]
            final_double_1.append(tmp_coord)
            
            tmp_x = j[0] - math.cos(vet_slope_arc)*0.8
            tmp_y = j[1] - math.sin(vet_slope_arc)*0.8
            tmp_coord = [tmp_x, tmp_y]
            final_double_2.append(tmp_coord)
            
        
        final_double_1 = np.array(final_double_1)
        final_double_2 = np.array(final_double_2)
        
        center_group1 = np.average(group1,axis=0)
        center_group2 = np.average(group2,axis=0)
        
        judge_dis_1 = point_distance_line(center_group1, final_double_1[0,:], final_double_1[2,:])
        judge_dis_2 = point_distance_line(center_group1, final_double_2[0,:], final_double_2[2,:])
        
        print(group1.shape)
        print(final_double_1[0,:])
        
        random1 = np.random.randn(final_double_1.shape[0], final_double_1.shape[1])*0.5
        random2 = np.random.randn(final_double_2.shape[0], final_double_2.shape[1])*0.5
        
        if judge_dis_1 > judge_dis_2:
            
            double_border_coord[i]=[final_double_2+random2, final_double_1+random1]
        else:
            double_border_coord[i]=[final_double_1+random1,final_double_2+random2]
            
        
    return center_point_1, border_line, double_border_coord


right_cell_order = pd.read_csv("right_oder_cellid.csv")["x"]
right_cell_order = np.array(right_cell_order)


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
            

cell_id = pd.read_csv("right_oder_cellid.csv")
cell_id = cell_id['0']

unique_cell_id = np.unique(cell_id)

cell_id_num = cell_id.copy()

dict_id_num = dict(zip(cell_id, cell_id_num))

for i in range(len(unique_cell_id)):
    tmp = unique_cell_id[i]
    index = np.where(cell_id == tmp)[0]
    cell_id_num[index] = i
    

colors = ["#FF8C00", "#32CD32", "#191970", "#00FA9A", "#7B68EE", "#FFD700", "#F08080", 
          "#FF0000", "#C71585", "#ffff14", "#00FFFF", "#680018", "#caa0ff"]

color_map = np.array(cell_id.copy())

for i in range(len(unique_cell_id)):
    index = np.where(cell_id == unique_cell_id[i])[0]
    color_map[index] = colors[i]
    
    

count_mat = mmread('normalized_data.mtx')
count_mat = count_mat.A
count_mat = count_mat.T
pca = PCA(n_components=30)
pca_mat = pca.fit_transform(count_mat)


P = sp_tsne.calculate_high_P(pca_mat,perplexity=50)


group = ["DE+NP"]
re = run_sp_tsne(P,pca_mat,1.1806615776081425e-04,group,cell_id)
re_sp, new_re_sp, new_cell_id, correct_double_dic,doublelet_half_1 = re
rr = get_final_double(group, new_cell_id, re_sp,correct_double_dic,doublelet_half_1)

final_double_coordinate = rr[-1]
plt.figure(None,(20,15))

for i in range(len(unique_cell_id)):
    
    tmp = np.where(new_cell_id == unique_cell_id[i])[0]
    plt.scatter(new_re_sp[tmp,0], new_re_sp[tmp,1],  s=50, label=unique_cell_id[i],c = colors[i]  )



plt.scatter(final_double_coordinate["DE+NP"][0][:,0], final_double_coordinate["DE+NP"][0][:,1], s=50,c='black')
plt.scatter(final_double_coordinate["DE+NP"][1][:,0], final_double_coordinate["DE+NP"][1][:,1], s=50,c='grey')
plt.axis("off")
plt.legend(markerscale=2, ncol= 4,prop={'size': 14})

######
#new_re_sp: coordinates of sp_tsne excluding doublelet cells
#new_cell_id : cell_id for sp_tsne excluding doublelet cells, same order as new_re_sp
#final_double_coordinate: a dict, key is "DE+NP" or other doublelet combination
#the first coordinates matrix is for the first group of "DE+NP" (DE)
#the second coordinates matrix is for the second group of "DE_NP"(NP)

#correct_double_dic:  a dict, key is  "DE+NP" or other doublelet combination, values are
#index of doublelets. the first index matrix is for the first group of "DE+NP" (DE)

#Take an example:
#See doublelet index of "DE"
print(correct_double_dic["DE+NP"][0])
#See doublelet coordinates of "DE"
print(final_double_coordinate["DE+NP"][0])
