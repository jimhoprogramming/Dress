import numpy as np
from matplotlib import path


def create_obj():
    #
    target = []
    target.append([1, 1, -1])
    target.append([-2, -2, 2])
    target.append([1, -1, 2])
    #
    p = []
    p.append([100, 100, 0.84999])
    return np.array(target), np.array(p)

def check_p_at_face_1(target,p):
    u,v = 0,0
    #Area = 0.5 *(-p1y*p2x + p0y*(-p1x + p2x) + p0x*(p1y - p2y) + p1x*p2y)

    #u = 1/(2*Area)*(p0y*p2x - p0x*p2y + (p2y - p0y)*px + (p0x - p2x)*py)
    #v = 1/(2*Area)*(p0x*p1y - p0y*p1x + (p0y - p1y)*px + (p1x - p0x)*py)
    
    if u > 0 and v >0 and u+v<1:
        return True
    else:
        return False

def check_p_at_face_2(target, p):
    print(target[:,:2].shape)
    pp = path.Path(target[:,:2])
    return pp.contains_points(np.reshape(p[0,:2],(1,2)))[0]

    
def face(target, p):
    print('target = \n {}'.format(target))
    ab = target[1,:] - target[0,:]
    print('ab = {}'.format(ab))
    ac = target[2,:] - target[0,:]
    print('ac = {}'.format(ac))
    n = np.cross(ab,ac)
    #print(type(n))
    #print(n.shape)
    print(u'法向量n = {}'.format(n))
    if p.shape[1] > 2:
        rel = np.dot(p - target[0,:], n)
        if rel <= -5.e-05 :
            rel = 0
    else:
    #  n[0] * (x - target[0,0]) + n[1] * (y - tartget[0,1]) + n[2] * (z_deep - target[0,2]) = 0 
        rel = -1 * (np.dot(p[0,:2] - target[0,:2], n[:2]) / n[2] - target[0,2])
    return rel
                 
if __name__ == '__main__':
    t,p = create_obj()
    #print(face(t,p))
    print(check_p_at_face_2(t,p))
