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

url = 'd://Dress//Data//base.obj'
vertices = []
my_vbo = None
windws_id = None
my_shader = None
View = None
Model = None
Project = None


def drawFunc():
    #清除之前画面
    glClear(GL_COLOR_BUFFER_BIT)
    glRotatef(0.1, 5, 5, 0)   #(角度,x,y,z)
    glutWireTeapot(0.5)
    #刷新显示
    glFlush()

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
##    vertices = np.array([[ 0, 1, 0 ],
##                        [ -1,-1, 0 ], 
##                        [ 1,-1, 0 ], 
##                        [ 2,-1, 0 ], 
##                        [ 4,-1, 0 ], 
##                        [ 4, 1, 0 ], 
##                        [ 2,-1, 0 ], 
##                        [ 4, 1, 0 ],
##                        [ 2, 1, 0 ],],'f')    
    return vertices,normals,textcoords,faces

# 提取由区域分割得到的人的点阵列表数据
def set_vertices():
    url = 'd://Dress//Data//vertex_output.npy'
    global vertices
    vertices = np.load(url)
    

# 初始化
def init():
    global my_vbo
    global vertices
    global my_shader
    global View, Model, Project
    print(vertices)
    # 设定颜色
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glColor4f(1.0, 1.0, 0.0, 0.0)
    # 准备顶点
    load_obj()
    #set_vertices()
    ## 准备着色器
    View = glm.translate(glm.mat4(), glm.vec3(-0.5, -1.2, 0.0))
    Model = glm.scale(glm.mat4(), glm.vec3(0.08, 0.08, 0.08))
    Project = glm.mat4()
    # 建立着色器的源代码
    # 编译着色器得到着色器对像
    VERTEX_SHADER = shaders.compileShader(
        """#version 120
        uniform mat4 View, Model, Project;
        in vec3 Vertex3;
        void main() {
            gl_PointSize = 0.5;
            //gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
            gl_Position = View * Model * Project * gl_Vertex;
            //gl_Position = ftransform(); 
        }""", GL_VERTEX_SHADER)
    FRAGMENT_SHADER = shaders.compileShader(
        """#version 120 
        void main() { 
            gl_FragColor = vec4( 0, 1, 0, 1 ); 
        }""", GL_FRAGMENT_SHADER)
    # 建立着色器程序
    
    # 关联着色器程序与着色器对像
    my_shader = shaders.compileProgram(VERTEX_SHADER,FRAGMENT_SHADER)   
    
    # 建立缓存
    # 关联变量
    print(vertices[0:4])
    my_vbo = vbo.VBO(np.asarray(vertices))
    print(my_vbo.size)

    return True

# 显示部分
def display():
    global my_vbo
    global my_shader
    global View, Model, Project
    # 清窗口缓存
    glClear(GL_COLOR_BUFFER_BIT)
    # 应用着色器程序
    shaders.glUseProgram(my_shader)
    glUniformMatrix4fv(glGetUniformLocation(my_shader,'View'), 1, GL_FALSE, glm.value_ptr(View))
    glUniformMatrix4fv(glGetUniformLocation(my_shader,'Model'), 1, GL_FALSE, glm.value_ptr(Model))
    glUniformMatrix4fv(glGetUniformLocation(my_shader,'Project'), 1, GL_FALSE, glm.value_ptr(Project))
    try: 
        # 绑定数组
        # 邦定缓存对象
        my_vbo.bind()
        glEnableClientState(GL_VERTEX_ARRAY)
        #glVertexPointer(3, GL_FLOAT, 0, my_vbo)
        glVertexPointerf(my_vbo)
        # 实行绘制命令
        #glDrawArrays(GL_TRIANGLES, 0, 9)
        #glDrawArrays(GL_LINE_LOOP, 0, 9)
        #glDrawArrays(GL_LINES, 0, 9)
        glDrawArrays(GL_POINTS, 0, 129)
    finally:
        glDisableClientState(GL_VERTEX_ARRAY)
        my_vbo.unbind()
        # 清除所有着色器
        shaders.glUseProgram(0)
    #刷新显示
    glFlush()
        

# 第三方窗口初始化
def init_window():
    global window_id
    # 初始化
    glutInit()
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
    if buttom == GLUT_RIGHT_BUTTON:
        glutDestroyWindow(window_id)
    
    

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
    

def draw_points():
    #清楚之前画面
    glClear(GL_COLOR_BUFFER_BIT)
    #glRotatef(-0.1, 5, 5, 0)   #(角度,x,y,z)
    #glPointSize(5.0)
    #glColor3f(1.0, 1.0, 0.0)
    glLoadIdentity ()
    glMatrixMode(GL_MODELVIEW)
    #gluLookAt(-10.0, -10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    #glViewport(-10, -10, 800, 600)
    gluOrtho2D(-20.0, 10.0, -10.0, 10.0) 
    glBegin(GL_TRIANGLES)
    for v in vertices:
        glVertex3f(v[0],v[1],v[2])
    glEnd()
    #glRotatef(0.1, 5, 5, 0)
    #刷新显示
    glFlush()    
        
def main_old(url):

    #使用glut初始化OpenGL
    print(OpenGL.__version__)
    glutInit()
    #显示模式:GLUT_SINGLE无缓冲直接显示|GLUT_RGBA采用RGB(A非alpha)
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
    #窗口位置及大小-生成
    glutInitWindowPosition(0,0)
    glutInitWindowSize(800,600)
    glutCreateWindow(b"first")
    # load obj
    vertices,normals,textcoords,faces = load_obj(url) 
    #调用函数绘制图像
    #print(vertices)
    glutDisplayFunc(draw_points)
    #glutDisplayFunc(drawFunc)
    #glutIdleFunc(draw_points)
    #主循环
    glutMainLoop()
    
if __name__== '__main__':
    
    main()
    #
    #load_obj(url = url)    
