# -*- coding: utf-8 -*-
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
import OpenGL
import numpy as np
import glm
#

url = '..//..//Data//base.obj'
##url = 'd://Dress//DataNormalize//check_base_obj.obj'
vertices = []
my_vbo = None
windws_id = None
Project = None
move_x = None
move_Z = None
move_y = None
r = 3.0
theata = 0
distance = 0

def load_obj():
    global vertices
    #vertices = []
    normals = []
    textcoords = []
    faces = []
    swapyz = False
    with open(file = url, mode = 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                #v = map(float, values[1:4])
                v=[ float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                vertices.append(v)
            elif values[0] == 'vn':
                v=[ float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                normals.append(v)
            elif values[0] == 'vt':
                v=[ float(x) for x in values[1:3]]
                textcoords.append(v)    
    return vertices,normals,textcoords,faces

# 提取由区域分割得到的人的点阵列表数据
def set_vertices():
    url = 'd://Dress//Data//vertex_output.npy'
    global vertices
    vertices = np.load(url)
    
# 初始化
def init():
    global vertices
    global move_x, move_z, move_y, r, theata
    print(vertices)
    # 设定颜色
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glColor4f(1.0, 1.0, 0.0, 0.0)
    # 准备顶点
    load_obj()
    #
    theata = 0
    move_x = r * np.cos(np.pi * theata)
    move_z = r * np.sin(np.pi * theata)

# 第三方窗口初始化
def init_window():
    global window_id
    # 初始化
    glutInit()
    # 显示模式:GLUT_SINGLE无缓冲直接显示|GLUT_RGBA采用RGB(A非alpha)
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
    # 窗口大小
    glutInitWindowSize(800,600)
    # 窗口位置
    glutInitWindowPosition(0,0)
    # 建立第一个窗口
    window_id = glutCreateWindow("first")
    # 设定视点
    glViewport(0, 0, 800, 600)
    return True

# 关于鼠标
def mouse_event(buttom, state, x, y):
    global window_id
    global move_x, move_z, move_y, r, theata
    if buttom == GLUT_RIGHT_BUTTON:
        glutDestroyWindow(window_id)
    elif buttom == GLUT_LEFT_BUTTON:
        theata += 1/40
        move_x = r * np.cos(np.pi * theata)
        move_z = r * np.sin(np.pi * theata)
        print(u'旋转x = {}, z = {}'.format(move_x, move_z))
        print(u'旋转 angle = {}pi'.format(theata * np.pi))
        glutPostRedisplay()
    elif buttom == 3:
        print('upping')
        r += 1/40
        move_x = r * np.cos(np.pi * theata)
        move_z = r * np.sin(np.pi * theata)
        print('upping r = {}'.format(r))
        glutPostRedisplay()
    elif buttom == 4:
        r -= 1/40
        move_x = r * np.cos(np.pi * theata)
        move_z = r * np.sin(np.pi * theata)
        print('down r = {}'.format(r))
        glutPostRedisplay()
        
# 显示部分
def display():
    global vertices
    global move_x, move_z, move_y
    #清之前画面
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity ()
    glMatrixMode(GL_MODELVIEW)
    #gluLookAt(move_x, 0.0, move_z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    #gluLookAt(-10.0, -10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    #glViewport(-10, -10, 800, 600)
    gluOrtho2D(-20.0, 10.0, -10.0, 10.0)
    #glOrtho(move_x, 1, move_z, 1, distance + 2, distance - 5)
    #glFrustum(-1, 1, -1, 1, 1 + distance, -1)
    glBegin(GL_POINTS)
    for v in vertices:
        glVertex3f(v[0],v[1],v[2])
    glEnd()
    #glRotatef(0.1, 5, 5, 0)
    #刷新显示
    glFlush()    
        
def main():
    print(OpenGL.__version__)
    # 第三方库窗口工作
    init_window()
    # 初始化
    init()
    #### 循环当没按下关闭
    # 显示
    glutDisplayFunc(display)
    # 刷写缓存
    #glutSwapBuffers()
    # 鼠标、键盘状态检测
    glutMouseFunc(mouse_event)
    # 主循环
    glutMainLoop()

    # 程序退出
    return True
    
if __name__== '__main__':    
    main()
   
