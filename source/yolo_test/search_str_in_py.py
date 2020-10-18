#  -*- coding:utf-8 -*-
# 循环查询指定目录里面的xml文件是否具有指定字符串并给出url


import os
#from lxml import etree
import re

# loop gaven url file
def loop_dir(directory, find_string):
    for f in os.listdir(directory):
        url = directory + '//' + f
        if os.path.isdir(url):
            loop_dir(url, find_string)
        elif os.path.isfile(url):
            check_string(url, find_string)
    return True

# check string if in file or not 
def check_string(url, find_string):
    ret = False
    pattern = r'.*this.*'.replace('this',find_string)
    find_mechine = re.compile(pattern, flags = 0)
    if os.path.splitext(url)[-1] == '.py':
        try:
            file = open(file = url, mode = 'r', encoding = 'utf8')
            context_list = file.readlines()
            #print(context_list)
            for text in context_list:                
                result = find_mechine.search(text)
                if result is not None:
                    print(url)
        except Exception as e:
            print(e)
        else:
            file.close()
    return ret

# collect context
def get_context(xpath):
    if isinstance(xpath, etree._Element):
        for node in xpath:
            get_context(node)
    else:
        print(xpath.attrib)
    
    
if __name__ == '__main__':
    print('this main')
    url = 'c://Program Files//Blender Foundation//Blender//2.79//scripts'
    #string = 'text_dlg_avoid_msg'
    #string = '0x7f0d00d4'
    #string = '0x7f0d0062'
    string = 'unwrap'
    loop_dir(directory = url, find_string = string)
    #string = '@dimen/about_us_app_version_code_size'
    #url = 'd://tools//aznxzs//lib//apktool//Turbo_VPN//res//layout//activity_about.xml'
    #check_string(url = url, find_string = string)

