import pygame, time, random, copy, sys, os
import numberlib_v2 as numberlib
import numpy as np
import psutil
import smtplib
import ssl, webbrowser
import matplotlib.pyplot as plt
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class LSTMcell():

    def __init__(self, leng, in_leng, alpha):
        self.alpha = alpha
        self.leng = leng
        self.in_leng = in_leng
        self.e = 2.718282
        self.aw = [np.random.normal(scale=1/np.sqrt(leng))  for i in range (0, self.leng)]
        self.ow = [np.random.normal(scale=1/np.sqrt(leng))  for i in range (0, self.leng)]
        self.iw = [np.random.normal(scale=1/np.sqrt(leng))  for i in range (0, self.leng)]
        self.fw = [np.random.normal(scale=1/np.sqrt(leng))  for i in range (0, self.leng)]
        self.au = [np.random.normal(scale=1/np.sqrt(in_leng))  for i in range (0, self.in_leng)]
        self.ou = [np.random.normal(scale=1/np.sqrt(in_leng))  for i in range (0, self.in_leng)]
        self.iu = [np.random.normal(scale=1/np.sqrt(in_leng))  for i in range (0, self.in_leng)]
        self.fu = [np.random.normal(scale=1/np.sqrt(in_leng))  for i in range (0, self.in_leng)]
        self.ab = 0
        self.ob = 0
        self.ib = 0
        self.fb = 0
        self.state = [0]
        self.out = [0]
        self.dstate = 0

    def forward(self, prelayer, ext_layer):
        a = self.ab
        for inn in range (0, self.leng):                          #can't use i
            a += self.aw[inn] * prelayer[inn]
        for inn in range (0, self.in_leng):
            a += ext_layer[inn] * self.au[inn]
        a = self.tanh(a)
        o = self.ob
        for inn in range (0, self.leng):
            o += self.ow[inn] * prelayer[inn]
        for inn in range (0, self.in_leng):
            o += ext_layer[inn] * self.ou[inn]
        o = self.sigmoid(o)
        i = self.ib
        for inn in range (0, self.leng):
            i += self.iw[inn] * prelayer[inn]
        for inn in range (0, self.in_leng):
            i += ext_layer[inn] * self.iu[inn]
        i = self.sigmoid(i)
        f = self.fb
        for inn in range (0, self.leng):
            f += self.fw[inn] * prelayer[inn]
        for inn in range (0, self.in_leng):
            f += ext_layer[inn] * self.fu[inn]
        f = self.sigmoid(f)
        hold_state = self.state[len(self.state)-1] * f
        ai = a * i
        hold_state += ai
        self.state.append(hold_state)
        hold_out = self.tanh(hold_state)*o
        self.out.append(ext_layer)
        return hold_out

    def sigmoid(self, values):
        return(1/(1+(np.exp(-1*values))))

    def tanh(self, values):
        return(np.exp(2*values)-1)/(np.exp(2*values)+1)

    def de_sig(self, values):
        return values*(1-values)

    def de_tanh(self, values):
        return 1 - (values ** 2)

    def back(self, dPpost, k, values, preV, last_v):
        layer_d = k
        layer_d += dPpost
        a = self.ab
        for inn in range (0, self.leng):                          #can't use i
            a += self.aw[inn] * preV[inn]
        for inn in range (0, self.in_leng):
            a += self.out[len(self.out)-1][inn] * self.au[inn]
        a = self.tanh(a)
        o = self.ob
        for inn in range (0, self.leng):
            o += self.ow[inn] * preV[inn]
        for inn in range (0, self.in_leng):
            o += self.out[len(self.out)-1][inn] * self.ou[inn]
        o = self.sigmoid(o)
        i = self.ib
        for inn in range (0, self.leng):
            i += self.iw[inn] * preV[inn]
        for inn in range (0, self.in_leng):
            i += self.out[len(self.out)-1][inn] * self.iu[inn]
        i = self.sigmoid(i)
        f = self.fb
        for inn in range (0, self.leng):
            f += self.fw[inn] * preV[inn]
        for inn in range (0, self.in_leng):
            f += self.out[len(self.out)-1][inn] * self.fu[inn]
        f = self.sigmoid(f)

        layer_o = layer_d * self.tanh(self.state[len(self.state)-1])*self.de_sig(o)
        layer_s = layer_d * self.de_tanh(self.tanh(self.state[len(self.state)-1]))*o + self.dstate
        layer_a = layer_s * i*self.de_tanh(a)
        layer_i = layer_s * a*self.de_sig(i)
        layer_f = layer_s * self.state[len(self.state)-2]*self.de_sig(f)
        self.dstate = layer_s * f

        dow = [0 for innn in range (0, self.leng)]
        dou = [0 for innn in range (0, self.in_leng)]
        dpre = [0 for innn in range (0, self.leng)]
        dout = [0 for innn in range (0, self.in_leng)]
        for inn in range (0, self.leng):
            dow[inn] = layer_o * preV[inn]
            dpre[inn] += layer_o * self.ow[inn]
        for inn in range (0, self.in_leng):
            dout[inn] = layer_o * self.ou[inn]
            dou[inn] = last_v[inn] * layer_o
        dob = layer_o
        
        daw = [0 for innn in range (0, self.leng)]
        dau = [0 for innn in range (0, self.in_leng)]
        for inn in range (0, self.leng):
            daw[inn] = layer_a * preV[inn]
            dpre[inn] += layer_a * self.aw[inn]
        for inn in range (0, self.in_leng):
            dout[inn] += layer_a * self.au[inn]
            dau[inn] = last_v[inn] * layer_a
        dab = layer_a
        
        diw = [0 for innn in range (0, self.leng)]
        diu = [0 for innn in range (0, self.in_leng)]
        for inn in range (0, self.leng):
            diw[inn] = layer_i * preV[inn]
            dpre[inn] += layer_i * self.iw[inn]
        for inn in range (0, self.in_leng):
            dout[inn] += layer_i * self.iu[inn]
            diu[inn] = last_v[inn] * layer_i
        dib = layer_i
        
        dfw = [0 for innn in range (0, self.leng)]
        dfu = [0 for innn in range (0, self.in_leng)]
        #print(layer_f, self.state[len(self.state)-2])
        for inn in range (0, self.leng):
            dfw[inn] = layer_f * preV[inn]
            dpre[inn] += layer_f * self.fw[inn]
        for inn in range (0, self.in_leng):
            dout[inn] += layer_f * self.fu[inn]
            dfu[inn] = last_v[inn] * layer_f
        dfb = layer_f
        
        dw = [[0,0,0,0] for innn in range (0, self.leng)]
        du = [[0,0,0,0] for innn in range (0, self.in_leng)]
        for inn in range (0, self.leng):
            dw[inn] = [dow[inn],daw[inn],diw[inn],dfw[inn]]
        for inn in range (0, self.in_leng):
            du[inn] = [dou[inn],dau[inn],diu[inn],dfu[inn]]
        db = [dob,dab,dib,dfb]

        safe_state = []
        for inn in range (0, len(self.state)-1):
            safe_state.append(self.state[inn])
        self.state = safe_state

        safe_out = []
        for inn in range (0, len(self.out)-1):
            safe_out.append(self.out[inn])
        self.out = safe_out
        
        return dw, dpre, du, dout, db #update, dprevious, hid_update, layer_d, dB

    def set_links(self, connections):
        self.connect = connections

    def get_links(self):
        return self.connect

    def get_weights(self):
        return self.aw

    def return_w(self):
        return self.au

    def update(self, tdow, tdaw, tdiw, tdfw, tdob, tdab, tdib, tdfb, tdou, tdau, tdiu, tdfu):
        for i in range (0, self.leng):
            self.ow[i] -= self.alpha * tdow[i]
            self.aw[i] -= self.alpha * tdaw[i]
            self.iw[i] -= self.alpha * tdiw[i]
            self.fw[i] -= self.alpha * tdfw[i]

        for i in range (0, self.in_leng):
            self.ou[i] -= self.alpha * tdou[i]
            self.au[i] -= self.alpha * tdau[i]
            self.iu[i] -= self.alpha * tdiu[i]
            self.fu[i] -= self.alpha * tdfu[i]

        self.ob -= self.alpha * tdob
        self.ab -= self.alpha * tdab
        self.ib -= self.alpha * tdib
        self.fb -= self.alpha * tdfb


class RNNcell():

    def __init__(self, leng, in_leng, alpha):
        self.alpha = alpha
        self.leng = leng
        self.bias = 0
        self.weights = [np.random.normal(scale=1/np.sqrt(leng))  for i in range (0, self.leng)]#2 * np.random.randn() - 1
        self.in_leng = in_leng
        self.connect = []
        self.in_weights = [np.random.normal(scale=1/np.sqrt(leng))  for i in range (0, self.in_leng)]
        self.e = 2.718282

    def forward(self, prelayer, ext_layer):

        total = self.bias
        
        for i in range (0, self.leng):
            total += self.weights[i] * prelayer[i]

        for i in range (0, self.in_leng):
            total += self.in_weights[i] * ext_layer[i]
        
        state = self.sigmoid(total)

        return state

    def sigmoid(self, value):
        return(1/(1+(np.exp(-1*value))))

    def de_sig(self, value):
        return value * (1 - value)

    def return_w(self):
        return self.in_weights

    def back(self, dPpost, future, values, preV, last_v):
        dPp = dPpost * self.de_sig(values)
        
        layer_d = future
        layer_d += dPpost
        layer_d = layer_d * self.de_sig(values)
        #print(layer_d)

        hid_update = [0 for i in range (0, self.in_leng)]
        for i in range (0, self.in_leng):
            hid_update[i] = last_v[i] * layer_d

        dB = layer_d

        update = [0 for i in range (0, self.leng)]
        for i in range (0, self.leng):
            update[i] = preV[i] * layer_d

        dprevious = [0 for i in range (0, self.leng)]
        for i in range (0, self.leng):
            dprevious[i] = layer_d * self.weights[i]

        return update, dprevious, hid_update, layer_d, dB

    def update(self, dwi, hdwi, db):
        for i in range (0, self.leng):
            self.weights[i] -= self.alpha * dwi[i]

        for i in range (0, self.in_leng):
            self.in_weights[i] -= self.alpha * hdwi[i]

        self.bias -= self.alpha * db

    def set_links(self, connections):
        self.connect = connections

    def get_links(self):
        return self.connect

    def get_weights(self):
        return self.weights


class cell():

    def __init__(self, leng, alpha, sig):
        self.sig = sig # 1 = sigmoid, 0 = ReLU
        self.alpha = alpha
        self.leng = leng
        self.bias = 0
        self.weights = [np.random.normal(scale=1/np.sqrt(leng)) for i in range (0, self.leng)]
        self.e = 2.718282
        self.connect = []


    def forward(self, prelayer, ext_layer):

        total = self.bias
        
        for i in range (0, self.leng):
            total += self.weights[i] * prelayer[i]
        
        if self.sig == 1:
            state = self.sigmoid(total)
        else:
            state = self.ReLU(total)
        
        return state

    def sigmoid(self, value):
        return(1/(1+(self.e**(-1*value))))

    def ReLU(self, value):
        if value > 0:
            return value
        else:
            return 0

    def de_sig(self, value):
        return value * (1 - value)

    def de_ReLU(self, value):
        return self.ReLU(value)

    def return_w(self):
        return []

    def back(self, dPpost, future, values, preV, last_v):
        if self.sig == 1:
            dPp = dPpost * self.de_sig(values)
        else:
            dPp = dPpost * self.de_ReLU(values)

        update = [0 for i in range (0, self.leng)]
        for i in range (0, self.leng):
            update[i] = preV[i] * dPp

        dB = dPp

        dprevious = [0 for i in range (0, self.leng)]
        for i in range (0, self.leng):
            dprevious[i] = dPp * self.weights[i]

        return update, dprevious, [], future, dB

    def update(self, dwi, hdwi, db):
        for i in range (0, self.leng):
            self.weights[i] -= self.alpha * dwi[i]
        self.bias -= self.alpha * db

    def set_links(self, connections):
        self.connect = connections

    def get_links(self):
        return self.connect

    def get_weights(self):
        return self.weights


class network():
    
    def __init__(self, structure, recursion = [], LSTM = [], binary = True, alpha = 0.01625, bias = True, reg = 0):
        if type(structure[0]) == list:
            self.type = "CNN"
        else:
            self.type = "FFN"#############################################
        self.bi = binary
        if LSTM != []:
            self.LSTM = True
            self.recursion = LSTM
        else:
            self.LSTM = False
            self.recursion = recursion
        self.regularization = reg
        if len(self.recursion) > 0:
            self.bi_re = True
        else:
            self.bi_re = False
        self.alpha = alpha
        self.hardlock_d = False
        self.bias = bias
        self.sample = False
        self.action = ""
        if self.type == "CNN":
            strut = []
            layers = 1
            layer_m = 1
            structure_m = []
            self.grid_type = []
            for i in range (0, len(structure)):
                if len(structure[i]) == 3:
                    layer_m = layer_m*structure[i][2]
                    structure_m.append(structure[i])
                    self.grid_type.append(1)
                elif len(structure[i]) == 2:
                    structure_m.append([structure[i][0],structure[i][1],layer_m])
                    self.grid_type.append(1)
                elif len(structure[i]) == 1:
                    layer = 1
                    structure_m.append([structure[i][0],1,layer])
                    self.grid_type.append(0)
            self.master_structure = structure_m
            self.backup_structure = structure
            self.outline = []
            if len(structure[0]) == 3:
                layers = layers*structure[0][2]
                strut.append(structure[0][0]*structure[0][1]*layers)
                self.outline.append([structure[0][0],structure[0][1],layers])
            elif len(structure[0]) == 2:
                strut.append(structure[0][0]*structure[0][1]*layers)
                self.outline.append([structure[0][0],structure[0][1],layers])
            elif len(structure[0]) == 1:
                layers = 1
                strut.append(structure[0][0])
                self.outline.append([structure[0][0],layers])
            for i in range (1, len(structure)):
                if len(structure[i]) == 3:
                    layers = layers*structure[i][2]
                    strut.append((self.outline[i-1][0]+1-structure[i][0])*(self.outline[i-1][1]+1-structure[i][1])*layers)
                    self.outline.append([(self.outline[i-1][0]+1-structure[i][0]),(self.outline[i-1][1]+1-structure[i][1]),layers])
                elif len(structure[i]) == 2:
                    strut.append((self.outline[i-1][0]+1-structure[i][0])*(self.outline[i-1][1]+1-structure[i][1])*layers)
                    self.outline.append([(self.outline[i-1][0]+1-structure[i][0]),(self.outline[i-1][1]+1-structure[i][1]),layers])
                elif len(structure[i]) == 1:
                    layers = 1
                    strut.append(structure[i][0])
                    self.outline.append([structure[i][0],1,layers])
            
            self.structure = strut
            self.length = len(self.structure)
            self.leng = self.length-1

            self.nodes = []
            self.nodes.append([cell(0, alpha, 0) for i in range (0, self.structure[0])])
            for j in range (1, self.leng):
                if self.grid_type[j] == 0:
                    self.nodes.append([cell(self.structure[j-1], alpha, 0) for i in range (0, self.structure[j])])
                elif self.grid_type[j] == 1:
                    self.nodes.append([cell(self.master_structure[j][0]*self.master_structure[j][1], alpha, 0) for i in range (0, self.structure[j])])
            if self.bi == False:
                if self.grid_type[self.leng] == 0:
                    self.nodes.append([cell(self.structure[self.leng-1], alpha, 1) for i in range (0, self.structure[self.leng])])
                elif self.grid_type[self.leng] == 1:
                    self.nodes.append([cell(self.master_structure[self.leng][0]*self.master_structure[self.leng][1], alpha, 1) for i in range (0, self.structure[self.leng])])
            else:
                if self.grid_type[self.leng] == 0:
                    self.nodes.append([cell(self.structure[self.leng-1], alpha, 1) for i in range (0, self.structure[self.leng])])
                elif self.grid_type[self.leng] == 1:
                    self.nodes.append([cell(self.master_structure[self.leng][0]*self.master_structure[self.leng][1], alpha, 1) for i in range (0, self.structure[self.leng])])
            self.values = [[0 for i in range (0, self.structure[j])] for j in range (0, self.length)]
            layers = self.master_structure[0][2]
            for i in range (0, self.leng):
                if self.grid_type[i+1] == 0:
                    layers = 1
                    tobe = [j for j in range (0, self.outline[i][0]*self.outline[i][1]*self.outline[i][2])]
                    for j in range (0, self.structure[i+1]):
                        self.nodes[i+1][j].set_links(tobe)
                elif self.grid_type[i+1] == 1:
                    for la in range (0, self.master_structure[i+1][2]):
                        for y in range (0, self.outline[i+1][1]):
                            for x in range (0, self.outline[i+1][0]):
                                for la2 in range (0, self.outline[i][2]):
                                    tobev = []
                                    for y2 in range (0, self.master_structure[i+1][1]):
                                        for x2 in range (0, self.master_structure[i+1][0]):
                                            tobev.append(x+x2+(y+y2)*self.outline[i][1]+la2*self.outline[i][0]*self.outline[i][1])
                                    self.nodes[i+1][la2*self.master_structure[i+1][2]*self.outline[i+1][0]*self.outline[i+1][1]+la*(self.outline[i+1][0]*self.outline[i+1][1])+y*self.outline[i+1][1]+x].set_links(tobev) #AlfieApprovedCode
                    layers = layers*self.master_structure[i+1][2]
                for j in range (0, self.structure[i+1]):
                    if self.bi_re == True:
                        for LST in range (0, len(self.recursion)):
                            if i+1 == self.recursion[LST]:
                                leng = self.nodes[i+1][j].leng
                                cone = self.nodes[i+1][j].get_links()
                                if self.LSTM == True:
                                    self.nodes[i+1][j] = LSTMcell(leng, self.structure[i+1], self.alpha)
                                    self.nodes[i+1][j].set_links(cone)
                                else:
                                    self.nodes[i+1][j] = RNNcell(leng, self.structure[i+1], self.alpha)
                                    self.nodes[i+1][j].set_links(cone)
        elif self.type == "FFN":
            self.length = len(structure)
            self.structure = structure
            self.leng = self.length -1
            self.nodes = [0 for i in range (0, self.length)]
            self.nodes[0] = [cell(0, 0, 0) for i in range (0, self.structure[0])]
            for i in range (1, self.leng):
                self.nodes[i] = [cell(self.structure[i-1], self.alpha, 0) for j in range (0, self.structure[i])]
            self.nodes[self.leng] = [cell(self.structure[self.leng-1], self.alpha, 1) for j in range (0, self.structure[self.leng])]
            for i in range (0, len(self.recursion)):
                if self.LSTM == True:
                    self.nodes[self.recursion[i]] = [LSTMcell(self.structure[self.recursion[i]-1], self.structure[self.recursion[i]], self.alpha) for j in range (0, self.structure[self.recursion[i]])]
                else:
                    self.nodes[self.recursion[i]] = [RNNcell(self.structure[self.recursion[i]-1], self.structure[self.recursion[i]], self.alpha) for j in range (0, self.structure[self.recursion[i]])]
        self.hardlock_d = False
        self.values = [[0 for i in range (0, self.structure[j])] for j in range (0, self.length)]

    def learn(self, grid, answer):
        if self.bi_re == True:
            set_leng = len(grid)
            cost = [0 for i in range (0, set_leng)]
            guess = [0 for i in range (0, set_leng)]
            valu = [0 for i in range (0, set_leng+1)]
            valu[0] = self.generate_f()
            L2D = [0 for i in range (0, set_leng)]
            estimate = [False for i in range (0, set_leng)]
            ans = [[0 for i in range (0, len(answer[0]))] for j in range (0, set_leng)]
            for intt in range (0, set_leng):
                per = psutil.virtual_memory().percent
                if per > 90:
                    exit()
                L2D[intt], cost[intt], estimate[intt], valv, guess[intt] = self.forward_pass(grid[intt], answer[intt])
                valu[intt+1] = copy.deepcopy(valv)
                if self.hardlock_d == True:
                    self.update_display(valu[intt+1],0,0)
                for i in range (0, len(answer[0])):
                    if valu[intt+1][self.leng][i] >= 0.5:
                        ans[intt][i] = 1
                    else:
                        ans[intt][i] = 0

            self.resetV()

            future = self.generate_f()

            dw = [0 for i in range (0, set_leng)]
            db = [0 for i in range (0, set_leng)]
            hdw = [0 for i in range (0, set_leng)]
            
            for intt in range (set_leng-1, -1, -1):
                per = psutil.virtual_memory().percent
                if per > 90:
                    exit()
                future, dw[intt], hdw[intt], db[intt] = self.get_alterations(L2D[intt], grid[intt], future, valu[intt+1], valu[intt])
                future = copy.deepcopy(future)


            tdaw = [[[0 for i in range (0, self.structure[k-1])] for j in range (0, self.structure[k])] for k in range (1, self.length)]
            tdow = [[[0 for i in range (0, self.structure[k-1])] for j in range (0, self.structure[k])] for k in range (1, self.length)]
            tdiw = [[[0 for i in range (0, self.structure[k-1])] for j in range (0, self.structure[k])] for k in range (1, self.length)]
            tdfw = [[[0 for i in range (0, self.structure[k-1])] for j in range (0, self.structure[k])] for k in range (1, self.length)]
            tdw = [[[0 for i in range (0, self.structure[k-1])] for j in range (0, self.structure[k])] for k in range (1, self.length)]
            tdb = [[0 for j in range (0, self.structure[k])] for k in range (1, self.length)]
            tdfb = [[0 for j in range (0, self.structure[k])] for k in range (1, self.length)]
            tdib = [[0 for j in range (0, self.structure[k])] for k in range (1, self.length)]
            tdob = [[0 for j in range (0, self.structure[k])] for k in range (1, self.length)]
            tdab = [[0 for j in range (0, self.structure[k])] for k in range (1, self.length)]
            thdw = [[[0 for i in range (0, self.structure[k])] for j in range (0, self.structure[k])] for k in range (1, self.length)]
            tdau = [[[0 for i in range (0, self.structure[k])] for j in range (0, self.structure[k])] for k in range (1, self.length)]
            tdou = [[[0 for i in range (0, self.structure[k])] for j in range (0, self.structure[k])] for k in range (1, self.length)]
            tdiu = [[[0 for i in range (0, self.structure[k])] for j in range (0, self.structure[k])] for k in range (1, self.length)]
            tdfu = [[[0 for i in range (0, self.structure[k])] for j in range (0, self.structure[k])] for k in range (1, self.length)]

            for i in range (0, len(dw)):
                for j in range (1, len(dw[i])):
                    for k in range (0, self.structure[j]):
                        if type(dw[i][j][k][0]) == list:
                            if self.type == "CNN":
                                zelda = self.nodes[j][k].get_links()
                                for too in range (0, len(zelda)):
                                    tdow[j-1][k][zelda[too]] += dw[i][j][k][too][0]
                                    tdaw[j-1][k][zelda[too]] += dw[i][j][k][too][1]
                                    tdiw[j-1][k][zelda[too]] += dw[i][j][k][too][2]
                                    tdfw[j-1][k][zelda[too]] += dw[i][j][k][too][3]
                            else:
                                for too in range (0, self.structure[j-1]):
                                    tdow[j-1][k][too] += dw[i][j][k][too][0]
                                    tdaw[j-1][k][too] += dw[i][j][k][too][1]
                                    tdiw[j-1][k][too] += dw[i][j][k][too][2]
                                    tdfw[j-1][k][too] += dw[i][j][k][too][3]
                        else:
                            if self.type == "CNN":
                                zelda = self.nodes[j][k].get_links()
                                for too in range (0, len(zelda)):
                                    ##print(i, j, k, too, zelda[too])
                                    tdw[j-1][k][zelda[too]] += dw[i][j][k][too]
                            else:
                                for too in range (0, self.structure[j-1]):
                                    tdw[j-1][k][too] += dw[i][j][k][too]

            for i in range (0, len(hdw)):
                for j in range (0, len(hdw[i])):
                    if self.LSTM == True:
                        asa = False
                        for azz in range (0, len(self.recursion)):
                            if self.recursion[azz] == j:
                                asa = True
                        if asa == True:
                            for k in range (0, self.structure[j]):
                                for too in range (0, self.structure[j]):
                                    tdou[j-1][k][too] += hdw[i][j][k][too][0]
                                    tdau[j-1][k][too] += hdw[i][j][k][too][1]
                                    tdiu[j-1][k][too] += hdw[i][j][k][too][2]
                                    tdfu[j-1][k][too] += hdw[i][j][k][too][3]
                        else:
                            for k in range (0, self.structure[j]):
                                for too in range (0, len(hdw[i][j][k])):
                                    thdw[j-1][k][too] += hdw[i][j][k][too]
                    else:
                        for k in range (0, self.structure[j]):
                            for too in range (0, len(hdw[i][j][k])):
                                thdw[j-1][k][too] += hdw[i][j][k][too]

            if self.bias == True:
                for i in range (0, len(db)):
                    for j in range (1, len(db[i])):
                        for k in range (0, self.structure[j]):
                            if type(db[i][j][k]) == list:
                                pass
                            else:
                                tdb[j-1][k] += db[i][j][k]

            for i in range (0, len(db)):
                for j in range (1, len(db[i])):
                    for k in range (0, self.structure[j]):
                        if type(db[i][j][k]) == list:
                            tdob[j-1][k] += db[i][j][k][0]
                            tdab[j-1][k] += db[i][j][k][1]
                            tdib[j-1][k] += db[i][j][k][2]
                            tdfb[j-1][k] += db[i][j][k][3]

            self.bi_update_all(tdw, thdw, tdb, tdow, tdaw, tdiw, tdfw, tdob, tdab, tdib, tdfb, tdou, tdau, tdiu, tdfu)


            self.resetV()
            self.LSTM_reset()

            return cost, ans

        else:
            l2d = []
            cost = 0
            estimate = []
            valv = []
            guess = []
            ans = [0 for i in range (0, self.structure[self.leng])]
            L2D, cost, estimate, valv, guess = self.forward_pass(grid, answer)
            valu = copy.deepcopy(valv)
            if self.hardlock_d == True:
                self.update_display(valu,0,0)
            for i in range (0, len(answer)):
                if valu[self.leng][i] >= 0.5:
                    ans[i] = 1
                else:
                    ans[i] = 0
            blank_pre = self.generate_f()
            future = self.generate_f()
            future, dw, hdw, db = self.get_alterations(L2D, grid, future, valu, blank_pre)
            if self.type == "CNN":
                tdw = [[[0 for i in range (0, len(self.nodes[k][j].get_links()))] for j in range (0, self.structure[k])] for k in range (1, self.length)]
            else:
                tdw = [[[0 for i in range (0, self.structure[k-1])] for j in range (0, self.structure[k])] for k in range (1, self.length)]
            tdb = [[0 for j in range (0, self.structure[k])] for k in range (1, self.length)]
            thdw = [[[0 for i in range (0, self.structure[k])] for j in range (0, self.structure[k])] for k in range (1, self.length)]
            for j in range (1, len(dw)):
                for k in range (0, self.structure[j]):
                    if self.type == "CNN":
                        zelda = self.nodes[j][k].get_links()
                        for too in range (0, len(zelda)):
                            tdw[j-1][k][too] += dw[j][k][too]
                    else:
                        for too in range (0, self.structure[j-1]):
                            tdw[j-1][k][too] += dw[j][k][too]
            if self.bias == True:
                for j in range (1, len(db)):
                    for k in range (0, self.structure[j]):
                        tdb[j-1][k] += db[j][k]
            self.update_all(tdw, thdw, tdb)
            return cost, ans

    def est(self, grid):
        if self.bi_re == True:
            set_leng = len(grid)
            cost = [0 for i in range (0, set_leng)]
            guess = [0 for i in range (0, set_leng)]
            valu = [0 for i in range (0, set_leng+1)]
            valu[0] = self.generate_f()
            L2D = [0 for i in range (0, set_leng)]
            estimate = [False for i in range (0, set_leng)]
            ans = [[0 for i in range (0, self.structure[self.leng])] for j in range (0, set_leng)]
            for intt in range (0, set_leng):
                valv, guess[intt] = self.forward_pas(grid[intt])
                valu[intt+1] = copy.deepcopy(valv)
                for i in range (0, self.structure[self.leng]):
                    if valu[intt+1][self.leng][i] >= 0.5:
                        ans[intt][i] = 1
                    else:
                        ans[intt][i] = 0

            self.resetV()
            self.LSTM_reset()

            return ans

        else:
            l2d = []
            cost = 0
            estimate = []
            valv = []
            guess = []
            ans = [0 for i in range (0, self.structure[self.leng])]
            valv, guess = self.forward_pas(grid)
            valu = copy.deepcopy(valv)
            if self.hardlock_d == True:
                self.update_display(valu,0,0)
            for i in range (0, self.structure[self.leng]):
                if valu[self.leng][i] >= 0.5:
                    ans[i] = 1
                else:
                    ans[i] = 0
            return ans


    def forward_pass(self, grid, answer):

        copy_values = copy.deepcopy(self.values)
        
        for i in range (0, len(self.values[0])):
            self.values[0][i] = grid[i]

        for layer in range (1, self.length):
            for node in range (0, self.structure[layer]):
                if self.type == "CNN":
                    zelda = self.nodes[layer][node].get_links()
                    awakening = []
                    for k in range (0, len(zelda)):
                        awakening.append(self.values[layer-1][zelda[k]])
                    self.values[layer][node] = self.nodes[layer][node].forward(awakening, copy_values[layer])
                else:
                    self.values[layer][node] = self.nodes[layer][node].forward(self.values[layer-1], copy_values[layer])

        cost = 0
        for i in range (0, self.structure[self.leng]):
            if self.values[self.leng][i] == 0:
                cost += -1*(answer[i]*np.log(0.000001)+(1-answer[i])*np.log(0.999999))
            elif self.values[self.leng][i] == 1:
                cost += -1*(answer[i]*np.log(0.999999)+(1-answer[i])*np.log(0.000001))
            else:
                cost += -1*(answer[i]*np.log(self.values[self.leng][i])+(1-answer[i])*np.log(1-self.values[self.leng][i]))
    
        L2D = [0 for i in range (0, self.structure[self.leng])]
        for i in range (0, self.structure[self.leng]):
            try:
                L2D[i] = -1*((answer[i]/self.values[self.leng][i])-((1-answer[i])/(1-self.values[self.leng][i])))
            except:
                print("catch")
                if self.values[self.leng][i] == 0:
                    L2D[i] = -1*((answer[i]/0.000001)-((1-answer[i])/0.999999))
                else:
                    L2D[i] = -1*((answer[i]/0.999999)-((1-answer[i])/0.000001))

        if self.values[self.leng][0] < 0.5:
            guess = 0
        else:
            guess = 1

        if guess == answer[0]:
            est = True
        else:
            est = False
        
        return L2D, cost, est, copy.deepcopy(self.values), guess

    def forward_pas(self, grid):

        copy_values = copy.deepcopy(self.values)
        
        for i in range (0, len(self.values[0])):
            self.values[0][i] = grid[i]

        for layer in range (1, self.length):
            for node in range (0, self.structure[layer]):
                if self.type == "CNN":
                    zelda = self.nodes[layer][node].get_links()
                    awakening = []
                    for k in range (0, len(zelda)):
                        awakening.append(self.values[layer-1][zelda[k]])
                    self.values[layer][node] = self.nodes[layer][node].forward(awakening, copy_values[layer])
                else:
                    self.values[layer][node] = self.nodes[layer][node].forward(self.values[layer-1], copy_values[layer])

        if self.values[self.leng][0] < 0.5:
            guess = 0
        else:
            guess = 1
        
        return copy.deepcopy(self.values), guess

    def get_alterations(self, L2D, grid, future, valu, pre_valu):
        dw = [[[] for i in range (0, self.structure[j])] for j in range (0, len(self.structure))]
        db = [[[] for i in range (0, self.structure[j])] for j in range (0, len(self.structure))]
        dPpost = [[0 for i in range (0, self.structure[j])] for j in range (0, len(self.structure))]
        dPpre = [[0 for i in range (0, self.structure[j])] for j in range (0, len(self.structure))]
        pass_d = [[[] for i in range (0, self.structure[j])] for j in range (0, len(self.structure))]
        n_f = [[0 for i in range (0, self.structure[j])] for j in range (0, len(self.structure))]
        
        self.values = valu

        dPpost[self.leng] = L2D
        for layer in range (self.leng, 0, -1):
            total_in_lay = [[] for i in range (0, self.structure[layer])]
            in_alt = [0 for i in range (0, self.structure[layer])]
            if self.LSTM == True:
                asa = False
                for elephant in range (0, len(self.recursion)):
                    if self.recursion[elephant] == layer:
                        asa = True
                if asa == True:
                    for giratine in range (0, self.structure[layer]):
                        if type(future[layer][giratine]) == list:
                            for elephant in range (0, self.structure[layer]):
                                in_alt[elephant] += future[layer][giratine][elephant]
                else:
                    for in_node in range (0, self.structure[layer]):
                        total_in_lay[in_node] = copy.deepcopy(self.nodes[layer][in_node].return_w())
                    for elephant in range (0, len(total_in_lay[0])):
                        for girafe in range (0, self.structure[layer]):
                            in_alt[elephant] += total_in_lay[girafe][elephant] * future[layer][girafe]
            else:
                for in_node in range (0, self.structure[layer]):
                    total_in_lay[in_node] = copy.deepcopy(self.nodes[layer][in_node].return_w())
                for elephant in range (0, len(total_in_lay[0])):
                    for girafe in range (0, self.structure[layer]):
                        in_alt[elephant] += total_in_lay[girafe][elephant] * future[layer][girafe]
            for node in range (self.structure[layer]):
                if self.type == "CNN":
                    zelda = self.nodes[layer][node].get_links()
                    awakening = []
                    for k in range (0, len(zelda)):
                        awakening.append(self.values[layer-1][k])
                    dw[layer][node], dPpre[layer][node], pass_d[layer][node], n_f[layer][node], db[layer][node] = self.nodes[layer][node].back(dPpost[layer][node], in_alt[node], self.values[layer][node], awakening, pre_valu[layer])
                else:
                    dw[layer][node], dPpre[layer][node], pass_d[layer][node], n_f[layer][node], db[layer][node] = self.nodes[layer][node].back(dPpost[layer][node], in_alt[node], self.values[layer][node], self.values[layer-1], pre_valu[layer])
            if self.type == "CNN":
                for j in range (0, self.structure[layer]):
                    zelda = self.nodes[layer][j].get_links()
                    for k in range (0, len(zelda)):
                        dPpost[layer-1][zelda[k]] += dPpre[layer][j][k]
            else:
                for j in range (0, self.structure[layer]):
                    for k in range (0, self.structure[layer-1]):
                        dPpost[layer-1][k] += dPpre[layer][j][k]
        
        return n_f, dw, pass_d, db

    def update_all(self, dweight, h_dweight, dbias):
        for layer in range (1, self.length):
            for node in range (0, self.structure[layer]):
                self.nodes[layer][node].update(dweight[layer-1][node], h_dweight[layer-1][node], dbias[layer-1][node])

    def bi_update_all(self, dweight, h_dweight, dbias, tdow, tdaw, tdiw, tdfw, tdob, tdab, tdib, tdfb, tdou, tdau, tdiu, tdfu):

        for layer in range (1, self.length):
            if self.LSTM == True:
                asa = False
                for vuvuzale in range (0, len(self.recursion)):
                    if self.recursion[vuvuzale] == layer:
                        asa = True
                if asa == True:
                    for node in range (0, self.structure[layer]):
                        self.nodes[layer][node].update(tdow[layer-1][node],tdaw[layer-1][node],tdiw[layer-1][node],tdfw[layer-1][node],tdob[layer-1][node],tdab[layer-1][node],tdib[layer-1][node],tdfb[layer-1][node],tdou[layer-1][node],tdau[layer-1][node],tdiu[layer-1][node],tdfu[layer-1][node])
                else:
                    for node in range (0, self.structure[layer]):
                        self.nodes[layer][node].update(dweight[layer-1][node], h_dweight[layer-1][node], dbias[layer-1][node])
            else:
                for node in range (0, self.structure[layer]):
                    self.nodes[layer][node].update(dweight[layer-1][node], h_dweight[layer-1][node], dbias[layer-1][node])


    def resetV(self):
        self.values = [[0 for i in range (0, self.structure[j])] for j in range (0, len(self.structure))]

    def LSTM_reset(self):
        if self.LSTM == True:
            for i in range (0, len(self.recursion)):
                for j in range (0, self.structure[self.recursion[i]]):
                    self.nodes[self.recursion[i]][j].state = [0]
                    self.nodes[self.recursion[i]][j].out = [0]
                    self.nodes[self.recursion[i]][j].dstate = 0

    def generate_f(self):
        future = [[0 for i in range (0, self.structure[j])] for j in range (0, len(self.structure))]
        return future

    def save(self, name):
        file = open(self.get_path()+name+".cir", "w+")
        file.write("5.1\n")
        file.write(self.type+"\n")
        if self.bi == True:
            file.write(str(1)+"\n")
        else:
            file.write(str(0)+"\n")
        if self.bi_re == True:
            file.write(str(1)+"\n")
        else:
            file.write(str(0)+"\n")
        if self.LSTM == True:####################################################
            file.write(str(1)+"\n")
        else:
            file.write(str(0)+"\n")
        for i in range (0, len(self.recursion)):
            file.write(str(self.recursion[i])+"\n")
        file.write("\n")
        file.write(str(self.alpha)+"\n")
        if self.bias == True:
            file.write(str(1)+"\n")
        else:
            file.write(str(0)+"\n")
        if self.LSTM == True:
            if self.type == "CNN":
                for i in range (0, len(self.backup_structure)):
                    for j in range (0, len(self.backup_structure[i])):
                        file.write(str(self.backup_structure[i][j])+"\n")
                    file.write("\n")
                file.write("\n")
                for layer in range (0, self.length):
                    asa = False
                    for iis in range (0, len(self.recursion)):
                        if self.recursion[iis] == layer:
                            asa = True
                    if asa == True:
                        for node in range (0, self.structure[layer]):
                            for weight in range (0, self.nodes[layer][node].leng):
                                file.write(str(self.nodes[layer][node].ow[weight])+"\n")
                                file.write(str(self.nodes[layer][node].aw[weight])+"\n")
                                file.write(str(self.nodes[layer][node].iw[weight])+"\n")
                                file.write(str(self.nodes[layer][node].fw[weight])+"\n")
                            for weight in range (0, self.nodes[layer][node].in_leng):
                                file.write(str(self.nodes[layer][node].ou[weight])+"\n")
                                file.write(str(self.nodes[layer][node].au[weight])+"\n")
                                file.write(str(self.nodes[layer][node].iu[weight])+"\n")
                                file.write(str(self.nodes[layer][node].fu[weight])+"\n")
                            file.write(str(self.nodes[layer][node].ob)+"\n")
                            file.write(str(self.nodes[layer][node].ab)+"\n")
                            file.write(str(self.nodes[layer][node].ib)+"\n")
                            file.write(str(self.nodes[layer][node].fb)+"\n")
                    else:
                        for node in range (0, self.structure[layer]):
                            for weight in range (0, self.nodes[layer][node].leng):
                                file.write(str(self.nodes[layer][node].weights[weight])+"\n")
                            if self.bias == True:
                                file.write(str(self.nodes[layer][node].bias)+"\n")
            elif self.type == "FFN":
                for i in range (0, self.length):
                    file.write(str(self.structure[i])+"\n")
                file.write("\n")
                for layer in range (0, self.length):
                    asa = False
                    for iis in range (0, len(self.recursion)):
                        if self.recursion[iis] == layer:
                            asa = True
                    if asa == True:
                        for node in range (0, self.structure[layer]):
                            for weight in range (0, self.nodes[layer][node].leng):
                                file.write(str(self.nodes[layer][node].ow[weight])+"\n")
                                file.write(str(self.nodes[layer][node].aw[weight])+"\n")
                                file.write(str(self.nodes[layer][node].iw[weight])+"\n")
                                file.write(str(self.nodes[layer][node].fw[weight])+"\n")
                            for weight in range (0, self.nodes[layer][node].in_leng):
                                file.write(str(self.nodes[layer][node].ou[weight])+"\n")
                                file.write(str(self.nodes[layer][node].au[weight])+"\n")
                                file.write(str(self.nodes[layer][node].iu[weight])+"\n")
                                file.write(str(self.nodes[layer][node].fu[weight])+"\n")
                            file.write(str(self.nodes[layer][node].ob)+"\n")
                            file.write(str(self.nodes[layer][node].ab)+"\n")
                            file.write(str(self.nodes[layer][node].ib)+"\n")
                            file.write(str(self.nodes[layer][node].fb)+"\n")
                    else:
                        for node in range (0, self.structure[layer]):
                            for weight in range (0, self.nodes[layer][node].leng):
                                file.write(str(self.nodes[layer][node].weights[weight])+"\n")
                            if self.bias == True:
                                file.write(str(self.nodes[layer][node].bias)+"\n")

        else:
            if self.type == "CNN":
                for i in range (0, len(self.backup_structure)):
                    for j in range (0, len(self.backup_structure[i])):
                        file.write(str(self.backup_structure[i][j])+"\n")
                    file.write("\n")
                file.write("\n")
                for layer in range (0, self.length):
                    for node in range (0, self.structure[layer]):
                        for weight in range (0, self.nodes[layer][node].leng):
                            file.write(str(self.nodes[layer][node].weights[weight])+"\n")
                        if self.bias == True:
                            file.write(str(self.nodes[layer][node].bias)+"\n")
                if self.bi_re == True:
                    for rec in range (0, len(self.recursion)):
                        for node in range (0, self.structure[self.recursion[rec]]):
                            for in_weight in range (0, self.nodes[self.recursion[rec]][node].in_leng):
                                file.write(str(self.nodes[self.recursion[rec]][node].in_weights[in_weight])+"\n")
            elif self.type == "FFN":
                for i in range (0, self.length):
                    file.write(str(self.structure[i])+"\n")
                file.write("\n")
                for layer in range (0, self.length):
                    for node in range (0, self.structure[layer]):
                        for weight in range (0, self.nodes[layer][node].leng):
                            file.write(str(self.nodes[layer][node].weights[weight])+"\n")
                        if self.bias == True:
                            file.write(str(self.nodes[layer][node].bias)+"\n")
                if self.bi_re == True:
                    for rec in range (0, len(self.recursion)):
                        for node in range (0, self.structure[self.recursion[rec]]):
                            for in_weight in range (0, self.nodes[self.recursion[rec]][node].in_leng):
                                file.write(str(self.nodes[self.recursion[rec]][node].in_weights[in_weight])+"\n")
        file.close()

    def load(self, name):
        file = open(self.get_path()+name+".cir", "r")
        if str(file.readline().rstrip()) != "5.1":
            print("errors may occur as the file was saved using a different version of cirilla")
        typ = str(file.readline().rstrip())
        bi = bool(int(file.readline().rstrip()))
        bi_re = bool(int(file.readline().rstrip()))
        LSTM = bool(int(file.readline().rstrip()))
        cut = str(file.readline().rstrip())
        recursion = []
        while cut != "":
            recursion.append(int(cut))
            cut = str(file.readline().rstrip())
        alpha = float(file.readline().rstrip())
        bias = bool(int(file.readline().rstrip()))
        if LSTM == True:
            if typ == "CNN":
                structure = []
                sub_structure = []
                upp = str(file.readline().rstrip())
                while upp != "":
                    while upp != "":
                        sub_structure.append(int(upp))
                        upp = str(file.readline().rstrip())
                    structure.append(sub_structure)
                    sub_structure = []
                    upp = str(file.readline().rstrip())

                self.rewrite(structure, typ, bi, bi_re, recursion, LSTM, alpha, bias)
                
                for layer in range (0, self.length):
                    asa = False
                    for iis in range (0, len(self.recursion)):
                        if self.recursion[iis] == layer:
                            asa = True
                    if asa == True:
                        for node in range (0, self.structure[layer]):
                            for weight in range (0, self.nodes[layer][node].leng):
                                self.nodes[layer][node].ow[weight] = float(file.readline().rstrip())
                                self.nodes[layer][node].aw[weight] = float(file.readline().rstrip())
                                self.nodes[layer][node].iw[weight] = float(file.readline().rstrip())
                                self.nodes[layer][node].fw[weight] = float(file.readline().rstrip())
                            for weight in range (0, self.nodes[layer][node].in_leng):
                                self.nodes[layer][node].ou[weight] = float(file.readline().rstrip())
                                self.nodes[layer][node].au[weight] = float(file.readline().rstrip())
                                self.nodes[layer][node].iu[weight] = float(file.readline().rstrip())
                                self.nodes[layer][node].fu[weight] = float(file.readline().rstrip())
                            self.nodes[layer][node].ob = float(file.readline().rstrip())
                            self.nodes[layer][node].ab = float(file.readline().rstrip())
                            self.nodes[layer][node].ib = float(file.readline().rstrip())
                            self.nodes[layer][node].fb = float(file.readline().rstrip())
                    else:
                        for node in range (0, self.structure[layer]):
                            for weight in range (0, self.nodes[layer][node].leng):
                                self.nodes[layer][node].weights[weight] = float(file.readline().rstrip())
                            if self.bias == True:
                                self.nodes[layer][node].bias = float(file.readline().rstrip())
            elif typ == "FFN":
                structure = []
                upp = str(file.readline().rstrip())
                while upp != "":
                    structure.append(int(upp))
                    upp = str(file.readline().rstrip())

                self.rewrite(structure, typ, bi, bi_re, recursion, LSTM, alpha, bias)
                    
                for layer in range (0, self.length):
                    asa = False
                    for iis in range (0, len(self.recursion)):
                        if self.recursion[iis] == layer:
                            asa = True
                    if asa == True:
                        for node in range (0, self.structure[layer]):
                            for weight in range (0, self.nodes[layer][node].leng):
                                self.nodes[layer][node].ow[weight] = float(file.readline().rstrip())
                                self.nodes[layer][node].aw[weight] = float(file.readline().rstrip())
                                self.nodes[layer][node].iw[weight] = float(file.readline().rstrip())
                                self.nodes[layer][node].fw[weight] = float(file.readline().rstrip())
                            for weight in range (0, self.nodes[layer][node].in_leng):
                                self.nodes[layer][node].ou[weight] = float(file.readline().rstrip())
                                self.nodes[layer][node].au[weight] = float(file.readline().rstrip())
                                self.nodes[layer][node].iu[weight] = float(file.readline().rstrip())
                                self.nodes[layer][node].fu[weight] = float(file.readline().rstrip())
                            self.nodes[layer][node].ob = float(file.readline().rstrip())
                            self.nodes[layer][node].ab = float(file.readline().rstrip())
                            self.nodes[layer][node].ib = float(file.readline().rstrip())
                            self.nodes[layer][node].fb = float(file.readline().rstrip())
                    else:
                        for node in range (0, self.structure[layer]):
                            for weight in range (0, self.nodes[layer][node].leng):
                                self.nodes[layer][node].weights[weight] = float(file.readline().rstrip())
                            if self.bias == True:
                                self.nodes[layer][node].bias = float(file.readline().rstrip())
        else:
            if typ == "CNN":
                structure = []
                sub_structure = []
                upp = str(file.readline().rstrip())
                while upp != "":
                    while upp != "":
                        sub_structure.append(int(upp))
                        upp = str(file.readline().rstrip())
                    structure.append(sub_structure)
                    sub_structure = []
                    upp = str(file.readline().rstrip())

                self.rewrite(structure, typ, bi, bi_re, recursion, LSTM, alpha, bias)
                
                for layer in range (0, self.length):
                    for node in range (0, self.structure[layer]):
                        for weight in range (0, self.nodes[layer][node].leng):
                            self.nodes[layer][node].weights[weight] = float(file.readline().rstrip())
                        if self.bias == True:
                            self.nodes[layer][node].bias = float(file.readline().rstrip())
                if self.bi_re == True:
                    for rec in range (0, len(self.recursion)):
                        for node in range (0, self.structure[self.recursion[rec]]):
                            for in_weight in range (0, self.nodes[self.recursion[rec]][node].in_leng):
                                self.nodes[self.recursion[rec]][node].in_weights[in_weight] = float(file.readline().rstrip())
            elif typ == "FFN":
                structure = []
                upp = str(file.readline().rstrip())
                while upp != "":
                    structure.append(int(upp))
                    upp = str(file.readline().rstrip())

                self.rewrite(structure, typ, bi, bi_re, recursion, LSTM, alpha, bias)
                    
                for layer in range (0, self.length):
                    for node in range (0, self.structure[layer]):
                        for weight in range (0, self.nodes[layer][node].leng):
                            self.nodes[layer][node].weights[weight] = float(file.readline().rstrip())
                        if self.bias == True:
                            self.nodes[layer][node].bias = float(file.readline().rstrip())
                if self.bi_re == True:
                    for rec in range (0, len(self.recursion)):
                        for node in range (0, self.structure[self.recursion[rec]]):
                            for in_weight in range (0, self.nodes[self.recursion[rec]][node].in_leng):
                                self.nodes[self.recursion[rec]][node].in_weights[in_weight] = float(file.readline().rstrip())

    def rewrite(self, structure, typ, bi, bi_re, recursion, LSTM, alpha, bias):
        self.type = typ
        self.bi = bi
        self.recursion = recursion
        self.LSTM = LSTM
        self.bi_re = bi_re
        self.alpha = alpha
        self.hardlock_d = False
        self.bias = bias
        self.sample = False
        self.action = ""
        if self.type == "CNN":
            strut = []
            layers = 1
            layer_m = 1
            structure_m = []
            self.grid_type = []
            for i in range (0, len(structure)):
                if len(structure[i]) == 3:
                    layer_m = layer_m*structure[i][2]
                    structure_m.append(structure[i])
                    self.grid_type.append(1)
                elif len(structure[i]) == 2:
                    structure_m.append([structure[i][0],structure[i][1],layer_m])
                    self.grid_type.append(1)
                elif len(structure[i]) == 1:
                    layer = 1
                    structure_m.append([structure[i][0],1,layer])
                    self.grid_type.append(0)
            self.master_structure = structure_m
            self.backup_structure = structure
            self.outline = []
            if len(structure[0]) == 3:
                layers = layers*structure[0][2]
                strut.append(structure[0][0]*structure[0][1]*layers)
                self.outline.append([structure[0][0],structure[0][1],layers])
            elif len(structure[0]) == 2:
                strut.append(structure[0][0]*structure[0][1]*layers)
                self.outline.append([structure[0][0],structure[0][1],layers])
            elif len(structure[0]) == 1:
                layers = 1
                strut.append(structure[0][0])
                self.outline.append([structure[0][0],layers])
            for i in range (1, len(structure)):
                if len(structure[i]) == 3:
                    layers = layers*structure[i][2]
                    strut.append((self.outline[i-1][0]+1-structure[i][0])*(self.outline[i-1][1]+1-structure[i][1])*layers)
                    self.outline.append([(self.outline[i-1][0]+1-structure[i][0]),(self.outline[i-1][1]+1-structure[i][1]),layers])
                elif len(structure[i]) == 2:
                    strut.append((self.outline[i-1][0]+1-structure[i][0])*(self.outline[i-1][1]+1-structure[i][1])*layers)
                    self.outline.append([(self.outline[i-1][0]+1-structure[i][0]),(self.outline[i-1][1]+1-structure[i][1]),layers])
                elif len(structure[i]) == 1:
                    layers = 1
                    strut.append(structure[i][0])
                    self.outline.append([structure[i][0],1,layers])
            
            self.structure = strut
            self.length = len(self.structure)
            self.leng = self.length-1

            self.nodes = []
            self.nodes.append([cell(0, alpha, 0) for i in range (0, self.structure[0])])
            for j in range (1, self.leng):
                if self.grid_type[j] == 0:
                    self.nodes.append([cell(self.structure[j-1], alpha, 0) for i in range (0, self.structure[j])])
                elif self.grid_type[j] == 1:
                    self.nodes.append([cell(self.master_structure[j][0]*self.master_structure[j][1], alpha, 0) for i in range (0, self.structure[j])])
            if self.bi == False:
                if self.grid_type[self.leng] == 0:
                    self.nodes.append([cell(self.structure[self.leng-1], alpha, 1) for i in range (0, self.structure[self.leng])])
                elif self.grid_type[self.leng] == 1:
                    self.nodes.append([cell(self.master_structure[self.leng][0]*self.master_structure[self.leng][1], alpha, 1) for i in range (0, self.structure[self.leng])])
            else:
                if self.grid_type[self.leng] == 0:
                    self.nodes.append([cell(self.structure[self.leng-1], alpha, 1) for i in range (0, self.structure[self.leng])])
                elif self.grid_type[self.leng] == 1:
                    self.nodes.append([cell(self.master_structure[self.leng][0]*self.master_structure[self.leng][1], alpha, 1) for i in range (0, self.structure[self.leng])])
            self.values = [[0 for i in range (0, self.structure[j])] for j in range (0, self.length)]
            layers = self.master_structure[0][2]
            for i in range (0, self.leng):
                if self.grid_type[i+1] == 0:
                    layers = 1
                    tobe = [j for j in range (0, self.outline[i][0]*self.outline[i][1]*self.outline[i][2])]
                    for j in range (0, self.structure[i+1]):
                        self.nodes[i+1][j].set_links(tobe)
                elif self.grid_type[i+1] == 1:
                    for la in range (0, self.master_structure[i+1][2]):
                        for y in range (0, self.outline[i+1][1]):
                            for x in range (0, self.outline[i+1][0]):
                                for la2 in range (0, self.outline[i][2]):
                                    tobev = []
                                    for y2 in range (0, self.master_structure[i+1][1]):
                                        for x2 in range (0, self.master_structure[i+1][0]):
                                            tobev.append(x+x2+(y+y2)*self.outline[i][1]+la2*self.outline[i][0]*self.outline[i][1])
                                    self.nodes[i+1][la2*self.master_structure[i+1][2]*self.outline[i+1][0]*self.outline[i+1][1]+la*(self.outline[i+1][0]*self.outline[i+1][1])+y*self.outline[i+1][1]+x].set_links(tobev) #AlfieApprovedCode
                    layers = layers*self.master_structure[i+1][2]
                for j in range (0, self.structure[i+1]):
                    if self.bi_re == True:
                        for LST in range (0, len(self.recursion)):
                            if i+1 == self.recursion[LST]:
                                leng = self.nodes[i+1][j].leng
                                cone = self.nodes[i+1][j].get_links()
                                if self.LSTM == True:
                                    self.nodes[i+1][j] = LSTMcell(leng, self.structure[i+1], self.alpha)
                                    self.nodes[i+1][j].set_links(cone)
                                else:
                                    self.nodes[i+1][j] = RNNcell(leng, self.structure[i+1], self.alpha)
                                    self.nodes[i+1][j].set_links(cone)
        elif self.type == "FFN":
            self.length = len(structure)
            self.structure = structure
            self.leng = self.length -1
            self.nodes = [0 for i in range (0, self.length)]
            self.nodes[0] = [cell(0, 0, 0) for i in range (0, self.structure[0])]
            for i in range (1, self.leng):
                self.nodes[i] = [cell(self.structure[i-1], self.alpha, 0) for j in range (0, self.structure[i])]
            self.nodes[self.leng] = [cell(self.structure[self.leng-1], self.alpha, 1) for j in range (0, self.structure[self.leng])]
            for i in range (0, len(self.recursion)):
                if self.LSTM == True:
                    self.nodes[self.recursion[i]] = [LSTMcell(self.structure[self.recursion[i]-1], self.structure[self.recursion[i]], self.alpha) for j in range (0, self.structure[self.recursion[i]])]
                else:
                    self.nodes[self.recursion[i]] = [RNNcell(self.structure[self.recursion[i]-1], self.structure[self.recursion[i]], self.alpha) for j in range (0, self.structure[self.recursion[i]])]
        self.hardlock_d = False
        self.values = [[0 for i in range (0, self.structure[j])] for j in range (0, self.length)]

    def order(self, number):
        numer = number
        a = [i for i in range (0, numer)]
        b = [0 for i in range (0, numer)]

        for i in range (numer, 0, -1):
            cal = random.randint(0, i-1)
            redo = True
            cali = 0
            precal = -1
            while redo == True:
                cal += cali
                cali = 0
                redo = False
                for j in range (0, numer-i):
                    if b[j] <= a[cal]:
                        if precal != -1:
                            if b[j] > a[precal]:
                                cali += 1
                                redo = True
                        else:
                            cali += 1
                            redo = True
                precal = cal
            b[numer-i] = a[cal]
        return b

    def imput(self, x, y, scale):
        self.action = "imput"
        grid = [[0 for i in range (0,x)] for j in range (0,y)]
        win = pygame.display.set_mode((x*scale,y*scale))
        self.window = win
        self.imput_x = x
        self.imput_y = y
        self.scale = scale
        sactive = True
        keyist = 0
        while sactive == True:
            pygame.event.get()
            self.update_display(self.scale, grid, 0)
            keys = pygame.key.get_pressed()
            if keys[13] == 1:
                if keyist == 0:
                    keyist = 1
                    sactive = False
            else:
                keyist = 0
            
            click = pygame.mouse.get_pressed()
            location = pygame.mouse.get_pos()
            if click[0] == 1:
                xco = int(location[0]/self.scale)
                yco = int(location[1]/self.scale)
                if xco >= 28 or yco >= 28:
                    pass
                else:
                    grid[yco][xco] = 1
        gridco = [0 for i in range (0,self.imput_x*self.imput_y)]
        for i in range (0, self.imput_y):
            for j in range (0, self.imput_x):
                gridco[i*self.imput_x+j] = grid[j][i]
        return gridco

    def init_display(self):
        self.run_window = 1
        pygame.display.init()
        win = pygame.display.set_mode((140,100))
        self.window = win
        pygame.font.init()
        return win

    def close_display(self):
        self.run_window = 0

    def hardlock_display(self):
        self.hardlock_d = True
        self.run_window = 0
        win = pygame.display.set_mode((640,240), pygame.RESIZABLE)
        self.window = win
        self.dis_cache = []
    
    def sample_display(self):
        self.sample = True

    def update_display_cache(self, cache):
        self.dis_cache = cache

    def update_display(self, cost, number, maximum_set):
        if self.hardlock_d == True:
            if self.sample == False:
                changed = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit(0)
                    if event.type == pygame.VIDEORESIZE:
                        scrsize = event.size  # or event.w, event.h
                        screen = pygame.display.set_mode(scrsize,pygame.RESIZABLE)
                        changed = True
                x = self.window.get_width()
                y = self.window.get_height()
                spacer = x / (self.length+1)
                self.window.fill((255,255,255))
                for i in range (1, self.length):
                    spacer2 = y / (self.structure[i]+1)
                    spacer3 = y / (self.structure[i-1]+1)
                    for j in range (0, self.structure[i]):
                        weights = self.nodes[i][j].get_weights()
                        if self.type == "CNN":
                            velda = self.nodes[i][j].get_links()
                            for k in range (0, len(velda)):
                                if weights[k] > 0:
                                    grette = (0,255,0)
                                else:
                                    grette = (255,0,0)
                                pygame.draw.line(self.window, grette, (int(spacer*(i+1)), int(spacer2*(j+1))), (int(spacer*(i)), int(spacer3*(velda[k]+1))))
                        else:
                            for k in range (0, self.structure[i-1]):
                                if weights[k] * cost[i-1][k] == 0:
                                    if weights[k] > 0:
                                        grette = (0,255,0)
                                    else:
                                        grette = (255,0,0)
                                elif weights[k] * cost[i][j-1] > 0:
                                    grette = (255,0,0)
                                else:
                                    grette = (0,255,0)
                                pygame.draw.line(self.window, grette, (int(spacer*(i+1)), int(spacer2*(j+1))), (int(spacer*(i)), int(spacer3*(k+1))))
                for i in range (1, self.length):
                    spacer2 = y / (self.structure[i]+1)
                    for j in range (0, self.structure[i]):
                        pygame.draw.rect(self.window, (0,0,0),(int(spacer*(i+1))-1, int(spacer2*(j+1))-1,4,4))
                spacer2 = y / (self.structure[0]+1)
                for j in range (0, self.structure[0]):
                    pygame.draw.rect(self.window, (0,0,0),(int(spacer*(1))-1, int(spacer2*(j+1))-1,4,4))
                spacer2 = y / (self.structure[self.leng]+1)
                for i in range (0, len(self.dis_cache)):
                    if type(self.dis_cache[i]) == np.float64 or type(self.dis_cache[i]) == float:
                        numberlib.drawdecleft(self.dis_cache[i], 6, [spacer*self.length+4, i*18], self.window, 2)
                    elif type(self.dis_cache[i]) == int:
                        numberlib.drawleft(self.dis_cache[i], [spacer*self.length+4, i*18], self.window, 2)
                    else:
                        print("cirilla display cache error")
                pygame.display.update()
            else:
                max_sample = 10
                changed = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit(0)
                    if event.type == pygame.VIDEORESIZE:
                        scrsize = event.size  # or event.w, event.h
                        screen = pygame.display.set_mode(scrsize,pygame.RESIZABLE)
                        changed = True
                x = self.window.get_width()
                y = self.window.get_height()
                spacer = x / (self.length+1)
                self.window.fill((255,255,255))
                for i in range (1, self.length):
                    if self.structure[i] > max_sample:
                        spacer2 = y / (max_sample+1)
                        exp_i = max_sample
                    else:
                        spacer2 = y / (self.structure[i]+1)
                        exp_i = self.structure[i]
                    if self.structure[i-1] > max_sample:
                        spacer3 = y / (max_sample+1)
                        exp_1 = max_sample
                    else:
                        spacer3 = y / (self.structure[i-1]+1)
                        exp_1 = (self.structure[i-1])
                    for j in range (0, exp_i):
                        weights = self.nodes[i][j].get_weights()
                        if self.type == "CNN":
                            velda = self.nodes[i][j].get_links()
                            for k in range (0, len(velda)):
                                if velda[k] >= max_sample:
                                    pass
                                else:
                                    if weights[k] > 0:
                                        grette = (0,255,0)
                                    else:
                                        grette = (255,0,0)
                                    pygame.draw.line(self.window, grette, (int(spacer*(i+1)), int(spacer2*(j+1))), (int(spacer*(i)), int(spacer3*(velda[k]+1))))
                        else:
                            for k in range (0, exp_1):
                                if weights[k] * cost[i-1][k] == 0:
                                    if weights[k] > 0:
                                        grette = (0,255,0)
                                    else:
                                        grette = (255,0,0)
                                elif weights[k] * cost[i][j-1] > 0:
                                    grette = (255,0,0)
                                else:
                                    grette = (0,255,0)
                                pygame.draw.line(self.window, grette, (int(spacer*(i+1)), int(spacer2*(j+1))), (int(spacer*(i)), int(spacer3*(k+1))))
                for i in range (1, self.length):
                    if self.structure[i] > max_sample:
                        spacer2 = y / (max_sample+1)
                        exp_i = max_sample
                    else:
                        spacer2 = y / (self.structure[i]+1)
                        exp_i = self.structure[i]
                    for j in range (0, exp_i):
                        pygame.draw.rect(self.window, (0,0,0),(int(spacer*(i+1)), int(spacer2*(j+1)),4,4))
                if self.structure[0] > max_sample:
                    spacer2 = y / (max_sample+1)
                    exp_0 = max_sample
                else:
                    spacer2 = y / (self.structure[0]+1)
                    exp_0 = self.structure[0]
                for j in range (0, exp_0):
                    pygame.draw.rect(self.window, (0,0,0),(int(spacer*(1)), int(spacer2*(j+1)),4,4))
                if self.structure[self.leng] > max_sample:
                    spacer2 = y / (max_sample+1)
                    exp_l = max_sample
                else:
                    spacer2 = y / (self.structure[self.leng]+1)
                    exp_l = self.structure[self.leng]
                for i in range (0, len(self.dis_cache)):
                    if type(self.dis_cache[i]) == np.float64 or type(self.dis_cache[i]) == float:
                        numberlib.drawdecleft(self.dis_cache[i], 6, [spacer*self.length, i*18], self.window, 2)
                    elif type(self.dis_cache[i]) == int:
                        numberlib.drawleft(self.dis_cache[i], [spacer*self.length, i*18], self.window, 2)
                    else:
                        print("cirilla display cache error")
                pygame.display.update()           
        else:  # note the rest of this is from an older version and may no longer work
            print("note the update_display script is from an older version and may no longer work")
            if self.action == "learn":#cost, number, maximum_set
                pygame.event.get()
                self.window.fill((255,255,255))
                pygame.draw.rect(self.window, (0,250,0), (20, 40, int((number/maximum_set)*100), 20))
                numberlib.drawdec((number/maximum_set)*100, 3, (70,50), self.window, 2)
                numberlib.drawleft(number, (20,70), self.window, 2)
                numberlib.drawdecleft(cost, 5, (20,20), self.window, 2)
                pygame.display.update()
            elif self.action == "test":
                pygame.event.get()
                self.window.fill((255,255,255))
                pygame.draw.rect(self.window, (0,250,0), (20, 40, int((number/maximum_set)*100), 20))
                numberlib.draw(int((number/maximum_set)*100), (70,50), self.window, 2)
                numberlib.drawleft(number, (20,70), self.window, 2)
                numberlib.drawleft(int((cost/(number))*100), (20,20), self.window, 2)
                pygame.draw.rect(self.window, (0,0,0), (28,29,1,1))
                pygame.display.update()
            elif self.action == "imput":    #cost = the scale ////// number = grid ////// maximum_set = blank(stored as 0)
                self.window.fill((255,255,255))
                for x in range (0,self.imput_x+1):
                    pygame.draw.line(self.window, (0,0,0), (x*cost, 0),(x*cost, cost*self.imput_x))
                for y in range (0,self.imput_y+1):
                    pygame.draw.line(self.window, (0,0,0), (0, y*cost),(cost*self.imput_y, y*cost))
                for y in range (0,self.imput_y):
                    for x in range (0,self.imput_x):
                        if number[y][x] == 1:
                            pygame.draw.rect(self.window, (0,0,0), (x*cost, y*cost, cost, cost))
                pygame.display.update()
            else:
                pygame.event.get()
                self.window.fill((0,0,0))
                pygame.display.update()#

    def display_feed(self, data):
        buf = np.float32(data)
        buf = buf.reshape(1, self.master_structure[0][0], self.master_structure[0][1], 1)
        image = np.asarray(buf).squeeze()
        plt.imshow(image)
        plt.show()

    def input_se(self, strr, temp):
        pygame.font.init()
        font = pygame.font.Font(self.get_path()+'Roboto.ttf', 12)
        existance = True
        input_str = ""
        capital = False
        while existance == True:
            temp.fill((255,255,255))
            text_width, text_height = font.size(strr)
            temp.blit(font.render(strr, True, (0,0,0)), (int(100-(0.5*text_width)),25))
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_0:
                        if capital == False:
                            input_str += "0"
                        else:
                            input_str += ")"
                    if event.key == pygame.K_1:
                        if capital == False:
                            input_str += "1"
                        else:
                            input_str += "!"
                    if event.key == pygame.K_2:
                        if capital == False:
                            input_str += "2"
                        else:
                            input_str += '"'
                    if event.key == pygame.K_3:
                        if capital == False:
                            input_str += "3"
                        else:
                            input_str += "£"
                    if event.key == pygame.K_4:
                        if capital == False:
                            input_str += "4"
                        else:
                            input_str += "$"
                    if event.key == pygame.K_5:
                        if capital == False:
                            input_str += "5"
                        else:
                            input_str += "%"
                    if event.key == pygame.K_6:
                        if capital == False:
                            input_str += "6"
                        else:
                            input_str += "^"
                    if event.key == pygame.K_7:
                        if capital == False:
                            input_str += "7"
                        else:
                            input_str += "&"
                    if event.key == pygame.K_8:
                        if capital == False:
                            input_str += "8"
                        else:
                            input_str += "*"
                    if event.key == pygame.K_9:
                        if capital == False:
                            input_str += "9"
                        else:
                            input_str += "("
                    if event.key == pygame.K_q:
                        if capital == False:
                            input_str += "q"
                        else:
                            input_str += "Q"
                    if event.key == pygame.K_w:
                        if capital == False:
                            input_str += "w"
                        else:
                            input_str += "W"
                    if event.key == pygame.K_e:
                        if capital == False:
                            input_str += "e"
                        else:
                            input_str += "E"
                    if event.key == pygame.K_r:
                        if capital == False:
                            input_str += "r"
                        else:
                            input_str += "R"
                    if event.key == pygame.K_t:
                        if capital == False:
                            input_str += "t"
                        else:
                            input_str += "T"
                    if event.key == pygame.K_y:
                        if capital == False:
                            input_str += "y"
                        else:
                            input_str += "Y"
                    if event.key == pygame.K_u:
                        if capital == False:
                            input_str += "u"
                        else:
                            input_str += "U"
                    if event.key == pygame.K_i:
                        if capital == False:
                            input_str += "i"
                        else:
                            input_str += "I"
                    if event.key == pygame.K_o:
                        if capital == False:
                            input_str += "o"
                        else:
                            input_str += "O"
                    if event.key == pygame.K_p:
                        if capital == False:
                            input_str += "p"
                        else:
                            input_str += "P"
                    if event.key == pygame.K_a:
                        if capital == False:
                            input_str += "a"
                        else:
                            input_str += "A"
                    if event.key == pygame.K_s:
                        if capital == False:
                            input_str += "s"
                        else:
                            input_str += "S"
                    if event.key == pygame.K_d:
                        if capital == False:
                            input_str += "d"
                        else:
                            input_str += "D"
                    if event.key == pygame.K_f:
                        if capital == False:
                            input_str += "f"
                        else:
                            input_str += "F"
                    if event.key == pygame.K_g:
                        if capital == False:
                            input_str += "g"
                        else:
                            input_str += "G"
                    if event.key == pygame.K_h:
                        if capital == False:
                            input_str += "h"
                        else:
                            input_str += "H"
                    if event.key == pygame.K_j:
                        if capital == False:
                            input_str += "j"
                        else:
                            input_str += "J"
                    if event.key == pygame.K_k:
                        if capital == False:
                            input_str += "k"
                        else:
                            input_str += "K"
                    if event.key == pygame.K_l:
                        if capital == False:
                            input_str += "l"
                        else:
                            input_str += "L"
                    if event.key == pygame.K_z:
                        if capital == False:
                            input_str += "z"
                        else:
                            input_str += "Z"
                    if event.key == pygame.K_x:
                        if capital == False:
                            input_str += "x"
                        else:
                            input_str += "X"
                    if event.key == pygame.K_c:
                        if capital == False:
                            input_str += "c"
                        else:
                            input_str += "C"
                    if event.key == pygame.K_v:
                        if capital == False:
                            input_str += "v"
                        else:
                            input_str += "V"
                    if event.key == pygame.K_b:
                        if capital == False:
                            input_str += "b"
                        else:
                            input_str += "B"
                    if event.key == pygame.K_n:
                        if capital == False:
                            input_str += "n"
                        else:
                            input_str += "N"
                    if event.key == pygame.K_m:
                        if capital == False:
                            input_str += "m"
                        else:
                            input_str += "M"
                    if event.key == pygame.K_PERIOD:
                        if capital == False:
                            input_str += "."
                        else:
                            input_str += ">"
                    if event.key == pygame.K_COMMA:
                        if capital == False:
                            input_str += ","
                        else:
                            input_str += "<"
                    if event.key == pygame.K_LSHIFT:
                        if capital == False:
                            capital = True
                        else:
                            capital = False
                    if event.key == pygame.K_BACKSPACE:
                        input_str = input_str[0:len(input_str)-1]
                    if event.key == pygame.K_RETURN:
                        existance = False
            text_width, text_height = font.size(input_str)
            temp.blit(font.render(input_str, True, (0,0,0)), (int(100-(0.5*text_width)),125))
            temp.blit(font.render("caps lock: " + str(capital), True, (0,0,0)), (0,170))
            temp.blit(font.render("press left shift to toggle caps lock", True, (0,0,0)), (0,185))
            pygame.display.flip()
        pygame.quit()
        return input_str
        
    def init_mail(self):
        try:
            file = open(self.get_path()+"email.txt", "r+")
        except:
            file = open(self.get_path()+"email.txt", "w")
        try:
            adress = file.readline().rstrip()
        except:
            adress = ""
        if adress == "":
            pygame.display.init()
            temp = pygame.display.set_mode((200,200))
            adress = self.input_se("input email adress up to @:", temp)
            adress += "@"
            temp = pygame.display.set_mode((200,200))
            adress += self.input_se("input email adress after @:", temp)
            temp = pygame.display.set_mode((200,200))
            password = self.input_se("input password:", temp)
            file.write(adress)
            file.write("\n")
            file.write(password)
        else:
            password = file.readline().rstrip()
        self.adress = adress
        self.password = password

    def mail_send(self, mail, heading, body):
        access = True
        print(self.adress, self.password)
        flip = True
        while access == True:
            if True:
            #try:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                    server.login(self.adress, self.password)
                    message = MIMEMultipart()
                    message["From"] = self.adress
                    message["To"] = mail
                    message["Subject"] = heading
                    message.attach(MIMEText(body, "plain"))
                    message = message.as_string()
                    server.sendmail(self.adress, mail, message)
                    access = True
            #except:
             #   if flip == True:
              #      webbrowser.open('https://myaccount.google.com/lesssecureapps?pli=1&rapt=AEjHL4O4yRYwd1_DtlnTi27Jhns1rwEpX9_Pm35F71fmNwBy5CHiddVvQoJL6Yab4wGUUR2MQYGOq6mwlehIUFXfUgI8PTXoCQ')
               #     flip = False

    def get_path(self):
        path = os.path.realpath(__file__)
        ex = True
        while ex == True:
            if path[len(path)-1:len(path)] != '\\':
                path = path[0:len(path)-1]
            else:
                ex = False
        return path

class neat():

    def __init__(self, structure, offspring, prob = 0.01):
        self.structure = structure
        self.first = True
        self.offspring = offspring
        self.alpha = prob
        self.weights = [[] for i in range (0, self.offspring)]
        for i in range (0, self.offspring):
            self.weights[i] = [[np.random.normal(scale=1/np.sqrt(self.structure[0])) for i in range (0, self.structure[0])] for j in range (0, self.structure[1])]
    
    def predict(self, inp, no):
        out = [0 for i in range (0, self.structure[1])]
        for i in range (self.structure[1]):
            for j in range (self.structure[0]):
                out[i] += self.weights[no][i][j] * inp[j]
        for i in range (self.structure[1]):
            out[i] = self.sigmoid(out[i])
        return out
    
    def bi_generation(self, order):
        if order == [0 for i in range (0, self.offspring)]:
            if self.first == True:
                self.structure = structure
                self.first = True
                self.offspring = offspring
                self.alpha = prob
                self.weights = [[] for i in range (0, self.offspring)]
                for i in range (0, self.offspring):
                    self.weights[i] = [[np.random.normal(scale=1/np.sqrt(self.structure[0])) for i in range (0, self.structure[0])] for j in range (0, self.structure[1])]
                return
            else:
                self.weights = self.last_weights
                order = self.old_order
        if True:
            total = 0
            for i in range (0, self.offspring):
                total += order[i]
            weight2 = [[] for i in range (0, self.offspring)]
            for child in range (0, self.offspring-1):
                howey = random.uniform(0.0, total)
                copyy = -1
                for i in range (0, self.offspring):
                    if copyy == -1:
                        if order[i] < howey:
                            howey -= order[i]
                        else:
                            copyy = i
                weg = copy.deepcopy(self.weights[copyy])
                howey2 = random.uniform(0.0, total)
                copyy2 = -1
                catch = 0
                while copyy2 == -1 and catch < 10:
                    for i in range (0, self.offspring):
                        if copyy2 == -1:
                            if order[i] < howey2:
                                howey2 -= order[i]
                            else:
                                copyy2 = i
                    if copyy2 == copyy:
                        copyy2 = -1
                        catch += 1
                if catch >= 10 :
                    copyy2 = random.randint(0, self.offspring-2)
                    if copyy2 >= copyy:
                        copyy2 += 1
                weg2 = copy.deepcopy(self.weights[copyy2])
                total2 = 0
                for endd in range (0, self.structure[1]):
                    for inp in range (0, self.structure[0]):
                        weg[endd][inp] = (weg[endd][inp] - weg2[endd][inp]) ** 2
                        total2 += weg[endd][inp]
                howey3 = random.uniform(0.0, total2)
                copyy3 = -1
                copyy4 = -1
                for endd in range (0, self.structure[1]):
                    for inp in range (0, self.structure[0]):
                        if copyy3 == -1:
                            if weg[endd][inp] < howey3:
                                howey3 -= weg[endd][inp]
                            else:
                                copyy3 = endd
                                copyy4 = inp
                weight2[child] = copy.deepcopy(self.weights[copyy])
                dog = random.uniform(0.0, 1.0)
                if dog <= self.alpha:
                    weight2[child][copyy3][copyy4] += np.random.normal(scale=1/(3*np.sqrt(self.structure[0])))
            maxi = -1
            maxn = -1
            for i in range (0, self.offspring):
                if order[i] > maxi:
                    maxi = order[i]
                    maxn = i
            weight2[self.offspring-1] = copy.deepcopy(self.weights[maxn])
            self.weights = weight2
            self.old_order = order
            self.last_weights = copy.deepcopy(self.weights)
            self.first = False
        
    def generation(self, order):
        if order == [0 for i in range (0, self.offspring)]:
            if self.first == True:
                self.structure = structure
                self.first = True
                self.offspring = offspring
                self.alpha = prob
                self.weights = [[] for i in range (0, self.offspring)]
                for i in range (0, self.offspring):
                    self.weights[i] = [[np.random.normal(scale=1/np.sqrt(self.structure[0])) for i in range (0, self.structure[0])] for j in range (0, self.structure[1])]
                return
            else:
                self.weights = self.last_weights
                order = self.old_order
        if True:
            total = 0
            for i in range (0, self.offspring):
                total += order[i]
            weight2 = [[] for i in range (0, self.offspring)]
            for child in range (0, self.offspring-1):
                howey = random.uniform(0.0, total)
                copyy = -1
                for i in range (0, self.offspring):
                    if copyy == -1:
                        if order[i] < howey:
                            howey -= order[i]
                        else:
                            copyy = i
                weight2[child] = copy.deepcopy(self.weights[copyy])
                for endd in range (0, self.structure[1]):
                    for inp in range (0, self.structure[0]):
                        dog = random.uniform(0.0, 1.0)
                        if dog <= self.alpha:
                            weight2[child][endd][inp] = np.random.normal(scale=1/np.sqrt(self.structure[0]))
            maxi = -1
            maxn = -1
            for i in range (0, self.offspring):
                if order[i] > maxi:
                    maxi = order[i]
                    maxn = i
            weight2[self.offspring-1] = self.weights[maxn]
            self.weights = weight2
            self.old_order = order
            self.last_weights = self.weights
            self.first = False

    def sigmoid(self, value):
        return(1/(1+(np.exp(-1*value))))
