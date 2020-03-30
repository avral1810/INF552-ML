"""Aviral Upadhyay
Vandit Maheshwari"""

import numpy as np
import matplotlib.pyplot as plt



class FastMap():
    def __init__(self, object_pair=None, distance=None, dimension=0):
        self.o_pair = np.array(object_pair)
        self.dist = np.array(distance)
        self.k = int(dimension) 
        self.obj_k_d = np.array 
        self._col = 0 
        self._max_dist = float(0) 
        self._new_dist_set = np.array
        self._max_Oa = 0
        self._max_Ob = 0        
        self.obj_set = set() 

    def set_obj_set(self, object_pair):
        for _each_pair in object_pair:
            self.obj_set.add(_each_pair[0])
            self.obj_set.add(_each_pair[1]) 
            
        self.obj_k_d = np.zeros((len(self.obj_set), self.k))        
        return self.obj_set
        
    def choose_distance_objects(self, object_pair, distance_set):
        Oa = 0
        original_Oa = 0
        Ob = object_pair[0][0]
        dist_set = distance_set
        pre_max_dist = 0
        max_dist = 0
        b_get_max = False 
        
        if (self._col == 0):
            while b_get_max is False:
                Oa, max_dist = self.get_max_distance(object_pair, dist_set, Ob)
                if ( max_dist >= pre_max_dist and original_Oa != Oa and max_dist != 0):
                    pre_max_dist = max_dist
                    original_Oa = Ob
                    Ob = Oa  
                else:
                    Oa = original_Oa
                    max_dist = pre_max_dist
                    b_get_max = True
                                        
                    self._max_Oa = Oa
                    self._max_Ob = Ob
                    self._max_dist = max_dist
        else:
            
            Oa = self._max_Oa
            Ob = self._max_Ob
            max_dist = self._max_dist
                       
        return Oa, Ob, max_dist

    def get_max_distance (self, object_pair, distance_set, Obj):
        farthest_O = 0
        max_distance = 0
        for idx, _each_pair in enumerate(object_pair):
            if _each_pair[0] == Obj :
                if distance_set[idx] >= max_distance :
                    max_distance = distance_set[idx]
                    farthest_O = _each_pair[1]
                else:
                    pass
            elif _each_pair[1] == Obj :
                if distance_set[idx] >= max_distance :
                    max_distance = distance_set[idx]
                    farthest_O = _each_pair[0] 
                else:
                    pass                   
            else:
                pass
        
        return farthest_O, max_distance
    

    def get_distance(self, object_pair, distance_set, xi, xj):
        
        for idx, _each_pair in enumerate(object_pair):
            if (xi == xj) :
                return 0 
            elif (_each_pair[0] == xi and _each_pair[1] == xj) or (_each_pair[1] == xi and _each_pair[0] == xj) :
                return distance_set[idx]
            else:
                pass
    
    def calculate_projection_distance(self, object_pair, distance_set, Oa, Ob):
        
        _xi = 0
        dis_set = distance_set
        obj_pair = object_pair
        Dai = 0
        Dab = self._max_dist
        Dbi = 0
        
        for _idx, _i in enumerate(self.obj_set):
            Dai = self.get_distance(obj_pair, dis_set, Oa, _i)
            Dbi = self.get_distance(obj_pair, dis_set, Ob, _i)           
            _xi = (Dai**2 + Dab**2 - Dbi**2) / (2 * Dab)
            self.obj_k_d[_idx][int(self._col)]=_xi
        
        return self.obj_k_d
    
    def projection_on_hyper_plane(self, object_pair, distance_set, xi_set, col):
        
        _xi = 0
        obj_pair = object_pair
        new_dist_set =np.zeros((len(obj_pair), 1)) 
        _xi = xi_set
        col = col
        D_Oij = 0
        D_xij = 0
        max_dist = 0
        max_Oa = 0
        max_Ob = 0
                
        for _idx, _pair in enumerate(obj_pair) :
            D_Oij = self.get_distance(object_pair, distance_set, _pair[0], _pair[1])
            D_xij = abs(_xi[int(_pair[0]-1)][int(col)] - _xi[int(_pair[1]-1)][int(col)])
            new_dist_set[_idx] = np.sqrt((D_Oij**2)-(D_xij**2))
            
            if (new_dist_set[_idx] >= max_dist):
                max_dist = new_dist_set[_idx]
                max_Oa = _pair[0]
                max_Ob = _pair[1]
                  
        self._new_dist_set = new_dist_set
        self._max_dist = max_dist
        self._max_Oa = max_Oa
        self._max_Ob = max_Ob
        
        return self._new_dist_set        
    
    
    def execute(self, object_pair, distance, dimension):
        
        self.o_pair = object_pair
        self.dist = distance
        self.k = dimension
        Oa = 0
        Ob = 0
        max_dist = 0
        dist_set = np.array(distance)
        new_dist_set = np.array
        col = 0
        self.obj_set = self.set_obj_set(object_pair)
        
        while (col < self.k) :
            Oa, Ob, _max_dist = self.choose_distance_objects(self.o_pair, dist_set)
            if (_max_dist == 0) :
                break
            else:
                self.calculate_projection_distance(self.o_pair, dist_set, Oa, Ob)
                col += 1
                self._col = col
                if(col < self.k):
                    _new_dist_set = self.projection_on_hyper_plane(self.o_pair, dist_set, self.obj_k_d, col-1) 
                    
                    _dist_set = _new_dist_set
                else:
                    break       
                        
        return self.obj_k_d

    def getInputData(self,filename):
    
        _object_pair = np.array
        _distance = np.array
        
        _data = np.genfromtxt(filename, delimiter='\t')
        _object_pair =  np.array(_data[0:, 0:2])
        _distance = np.array(_data[0:,2])
        
        return _object_pair, _distance  

    def getObjectName(self,filename):
        _label_set = []
        
        with open(filename) as f:
            _label_set = f.read().splitlines()
        
        return _label_set  
    
    def plot(self, label_set):
        plt.xlabel("x-axis") 
        plt.ylabel("y-axis") 

        for i in range(len(self.obj_k_d)):
            _x = self.obj_k_d[i][0]
            _y = self.obj_k_d[i][1]
            _label = label_set[i]
            plt.scatter(_x,_y)
            plt.annotate(_label,(_x,_y))
        plt.show()