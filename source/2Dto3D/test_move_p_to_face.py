import numpy as np

def create_obj():
    #
    target = []
    target.append([1, 1, -1])
    target.append([-2, -2, 2])
    target.append([1, -1, 2])
    #
    p = []
    p.append([, 0.2, -0.5])
    return np.array(target), np.array(p)

def face(target, p):
    print('target = \n {}'.format(target))
    ab = target[1,:] - target[0,:]
    print('ab = {}'.format(ab))
    ac = target[2,:] - target[0,:]
    print('ac = {}'.format(ac))
    n = np.cross(ab,ac)
    print(u'法向量n = {}'.format(n))
    rel = np.dot(p - target[0,:], n)
    return rel
                 
if __name__ == '__main__':
    t,p = create_obj()
    print(face(t,p))
    
