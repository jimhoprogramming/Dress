# -*- coding: utf-8 -*-
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
import OpenGL  #'3.1.5'  GL3.1-GLSL140
import numpy as np
import glm
#

url = 'd://Dress//DataResize//base.obj'
##url = 'c://Users//FRONTIER//3D Objects//check_base_obj.obj'
##url = 'd://Dress//DataNormalize//check_base_obj.obj'
##url = 'd://Dress//DataNormalize//base.obj'

vertices = []
my_vbo = None
windws_id = None
my_shader = None
View = None
Model = None
Project = None
move_x = None
move_Z = None
move_y = None
r = 1
theata = 0
distance = 0

def normlize():
    global vertices 
    obj_min = np.min(vertices, axis = 0)
    obj_max = np.max(vertices, axis = 0)
    max_len = np.max(obj_max - obj_min)
    mul_vector = np.array([2,2,1]).reshape(1,3)
    sub_vector = np.array([1,1,0]).reshape(1,3)
    new_points = (vertices - obj_min) * mul_vector / max_len - sub_vector
    return new_points
    
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
                v=[ float(x) for x in values[1:]]
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
    '''
    vertices = np.array([[ 0, 1, 0 ],
                        [ -1,-1, 0 ], 
                        [ 1,-1, 0 ], 
                        [ 2,-1, 0 ], 
                        [ 4,-1, 0 ], 
                        [ 4, 1, 0 ], 
                        [ 2,-1, 0 ], 
                        [ 4, 1, 0 ],
                        [ 2, 1, 0 ],],'float32')
    '''
    vertices = np.array(vertices,'float32')
    print(u'对像基本信息：')
    print(type(vertices))
    print('nbytes:{}-{}'.format(vertices.nbytes,vertices.dtype))
    print('obj max = {},min = {}'.format(vertices.max(),vertices.min()))
    print('obj shape = {}'.format(vertices.shape))
    print(u'顶点坐标:{}'.format(vertices))
    return vertices

# 提取由区域分割得到的人的点阵列表数据
def set_vertices():
    url = 'd://Dress//Data//vertex_output.npy'
    global vertices
    vertices = np.load(url)
    
def my_callback(source, GLenum_type, GLuint_id, GLenum_serverity, GLsizei_length, message, userParam):
    print('source:{},type:{},id:{},serverty:{},message:{}'.format(source, GLenum_type, GLuint_id, GLenum_serverity, message))
    
# 初始化
def init():
    global my_vbo
    global vertices
    global my_shader
    global View, Model, Project
    global move_x, move_z, move_y, r, theata
    # 设定服务器的调试开启
##    userParam = glm.value_ptr(None)
##    glEnable(GL_DEBUG_OUTPUT)
##    glDebugMessageCallback(my_callback, userParam)
##    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE,GL_DONT_CARE, 0, userParam, GL_TRUE)
    # 设定颜色
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glColor4f(1.0, 1.0, 0.0, 0.0)
    # 准备顶点
    load_obj()
    # 准备中心点
    center_point = (np.max(vertices, axis = 0) + np.min(vertices, axis = 0)) / 2.0
    print(vertices)
    print(u'中心坐标：{}'.format(center_point))
    #vertices = (vertices - np.min(vertices, axis = 0)) / center_point - 0.1
    #print(u'位移到中心处的各顶点坐标：{}'.format(vertices))
    ## 准备着色器
    View = glm.translate(glm.mat4(), glm.vec3(0, 0, 0))
    #Model = glm.scale(glm.mat4(), glm.vec3(0.08, 0.08, 0.08))
    Model = glm.scale(glm.mat4(), glm.vec3(.3, .3, .3))
    #
##    Project = glm.frustum(-1, 1, -1, 1, 0, -1)
##    print(Project)
    #
##    Project = glm.ortho(-1.1, 1.1, -1.1, 1.1, 1.1, -1.1)
##    print(Project)
    #
    print('init angle = {}pi'.format(theata * np.pi))
    move_x = r * np.cos(np.pi * theata)
    move_z = r * np.sin(np.pi * theata)
    Project = glm.lookAt(glm.vec3(0, move_x, move_z), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0, 1, 0))
    #print(u'模拟计算着色器内部值：')
    #print(np.dot(np.concatenate([vertices, np.ones((vertices.shape[0],1))],1),Project))
    # 建立着色器的源代码
    # 编译着色器得到着色器对像
    VERTEX_SHADER = shaders.compileShader(
        """#version 140 core
        in vec3 Vertex3;
        uniform mat4 View, Model, Project;
        void main() {
            gl_PointSize = 1.0;
            //gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
            gl_Position = View * Model * Project * vec4(Vertex3, 1.0);
            //gl_Position = View * Model * Project * gl_Vertex;
            //gl_Position = ftransform(); 
        }""", GL_VERTEX_SHADER)
    FRAGMENT_SHADER = shaders.compileShader(
        """#version 140 core
        void main() {
            gl_FragColor = vec4( 0, 1, 0, 1 ); 
        }""", GL_FRAGMENT_SHADER)
    # 建立着色器程序
    
    # 关联着色器程序与着色器对像
    my_shader = shaders.compileProgram(VERTEX_SHADER,FRAGMENT_SHADER)   
    
    # 建立缓存
    # 关联变量
    #print('obj head 4 point: {}'.format(vertices[0:9,:]))
    #new_vertices = normlize()
    my_vbo = vbo.VBO(np.asarray(vertices))
    print('vbo size ={}'.format(my_vbo.size))

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
    index_number = (vertices.shape[0])
    try: 
        # 绑定数组
        # 邦定缓存对象
        my_vbo.bind()
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 12, my_vbo)
        #glVertexPointerf(my_vbo)
        #glVertexAttribPointer(0, 4, GL_FLOAT, False, 0, my_vbo)
        # 实行绘制命令
        #glDrawArrays(GL_TRIANGLES, 0, 9)
        #glDrawArrays(GL_TRIANGLE_STRIP, 0, index_number)
        #glDrawArrays(GL_LINE_LOOP, 0, 18)
        #glDrawArrays(GL_LINES, 0, 9)
        glDrawArrays(GL_POINTS, 0, index_number)
    finally:
        glDisableClientState(GL_VERTEX_ARRAY)
        my_vbo.unbind()
        # 清除所有着色器
        shaders.glUseProgram(0)
    #刷新显示
    glFlush()
    print('display ')
        

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
    glDepthRange(-2.0, 2.0)
    return True

# 关于鼠标
def mouse_event(buttom, state, x, y):
    #print(buttom,state)
    global window_id
    global Project
    global move_x, move_z, r,theata, distance
    if buttom == GLUT_RIGHT_BUTTON:
        glutDestroyWindow(window_id)
    elif buttom == GLUT_LEFT_BUTTON:
        theata += 1/40
        move_x = r * np.cos(np.pi * theata)
        move_z = r * np.sin(np.pi * theata)
##        print(u'旋转x = {}, z = {}'.format(move_x, move_z))
##        print(u'旋转 angle = {}pi'.format(theata * np.pi))
        Project = glm.lookAt(glm.vec3(0, move_x, move_z), glm.vec3(0, 0.0, 0.0), glm.vec3(0, 1, 0))
        glutPostRedisplay()
    elif buttom == 3:
##        print('upping')
        distance += 1/40
##        Project = glm.ortho(-1, 1, -1, 1, distance + 2, distance - 5)
        Project = glm.frustum(-1, 1, -1, 1, 1 + distance, -1)
        glutPostRedisplay()
    elif buttom == 4:
        distance -= 1/40
##        print('down')
##        Project = glm.ortho(-1, 1, -1, 1, distance + 2, distance - 5)
        Project = glm.frustum(-1, 1, -1, 1, 1 + distance, -1)
        glutPostRedisplay()
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
    glutSwapBuffers()
    # 鼠标、键盘状态检测
    glutMouseFunc(mouse_event)
    # 主循环
    glutMainLoop()
    # 程序退出
    return True
        
if __name__== '__main__':
    
    main()
    #
    #load_obj(url = url)    
