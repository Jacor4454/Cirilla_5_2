import pygame

def draw(number, co_ord, win, scale):
    rear = -(0.5*8*len(str(number)))
    for excess in range (0, len(str(number))):
        letter = []
        if str(number)[excess] == "1":
            letter = [[0,1,3,1],[3,2,1,6],[0,8,6,1]]
        elif str(number)[excess] == "2":
            letter = [[0,1,6,1],[5,2,1,3],[0,4,6,1],[0,4,1,4],[0,8,6,1]]
        elif str(number)[excess] == "3":
            letter = [[0,1,5,1],[5,1,1,8],[3,4,2,1],[0,8,5,1]]
        elif str(number)[excess] == "4":
            letter = [[0,1,1,5],[0,6,6,1],[3,5,1,1],[3,7,1,2]]
        elif str(number)[excess] == "5":
            letter = [[0,1,6,1],[0,2,1,2],[0,4,6,1],[5,5,1,3],[0,8,6,1]]
        elif str(number)[excess] == "6":
            letter = [[0,1,6,1],[0,2,1,7],[1,4,5,1],[1,8,5,1],[5,5,1,3]]
        elif str(number)[excess] == "7":
            letter = [[0,1,6,1],[5,2,1,7]]
        elif str(number)[excess] == "8":
            letter = [[0,1,6,1],[0,2,1,6],[5,2,1,6],[0,8,6,1],[1,4,4,1]]
        elif str(number)[excess] == "9":
            letter = [[0,1,6,1],[0,2,1,3],[5,2,1,6],[0,8,6,1],[1,4,4,1]]
        elif str(number)[excess] == "0":
            letter = [[0,1,6,1],[0,2,1,6],[5,2,1,6],[0,8,6,1]]
        for i in range (0,len(letter)):
            pygame.draw.rect(win, (0,0,200), (co_ord[0]+(rear+letter[i][0]+8*excess)*scale,co_ord[1]-5*scale+letter[i][1]*scale,letter[i][2]*scale,letter[i][3]*scale))

def drawdec(number, place, co_ord, win, scale):
    scale = int(scale)
    rear = -(0.5*8*(len(str(int(number)))+place))-1
    if place > len(str(number)) - 1 - len(str(int(number))):
        place = len(str(number)) - 1 - len(str(int(number)))
    for excess in range (0, len(str(int(number)))):
        letter = []
        if str(number)[excess] == "1":
            letter = [[0,1,3,1],[3,2,1,6],[0,8,6,1]]
        elif str(number)[excess] == "2":
            letter = [[0,1,6,1],[5,2,1,3],[0,4,6,1],[0,4,1,4],[0,8,6,1]]
        elif str(number)[excess] == "3":
            letter = [[0,1,5,1],[5,1,1,8],[3,4,2,1],[0,8,5,1]]
        elif str(number)[excess] == "4":
            letter = [[0,1,1,5],[0,6,6,1],[3,5,1,1],[3,7,1,2]]
        elif str(number)[excess] == "5":
            letter = [[0,1,6,1],[0,2,1,2],[0,4,6,1],[5,5,1,3],[0,8,6,1]]
        elif str(number)[excess] == "6":
            letter = [[0,1,6,1],[0,2,1,7],[1,4,5,1],[1,8,5,1],[5,5,1,3]]
        elif str(number)[excess] == "7":
            letter = [[0,1,6,1],[5,2,1,7]]
        elif str(number)[excess] == "8":
            letter = [[0,1,6,1],[0,2,1,6],[5,2,1,6],[0,8,6,1],[1,4,4,1]]
        elif str(number)[excess] == "9":
            letter = [[0,1,6,1],[0,2,1,3],[5,2,1,6],[0,8,6,1],[1,4,4,1]]
        elif str(number)[excess] == "0":
            letter = [[0,1,6,1],[0,2,1,6],[5,2,1,6],[0,8,6,1]]
        for i in range (0,len(letter)):
            pygame.draw.rect(win, (0,0,200), (co_ord[0]+(rear+letter[i][0]+8*excess)*scale,co_ord[1]-5*scale+letter[i][1]*scale,letter[i][2]*scale,letter[i][3]*scale))
    
    pygame.draw.rect(win, (0,0,200), (co_ord[0]+(rear+8*len(str(int(number))))*scale, co_ord[1]-5*scale+8*scale, 2*scale, 2*scale))
    
    for excess in range (0, place):
        letter = []
        if str(number)[excess+1+len(str(int(number)))] == "1":
            letter = [[0,1,3,1],[3,2,1,6],[0,8,6,1]]
        elif str(number)[excess+1+len(str(int(number)))] == "2":
            letter = [[0,1,6,1],[5,2,1,3],[0,4,6,1],[0,4,1,4],[0,8,6,1]]
        elif str(number)[excess+1+len(str(int(number)))] == "3":
            letter = [[0,1,5,1],[5,1,1,8],[3,4,2,1],[0,8,5,1]]
        elif str(number)[excess+1+len(str(int(number)))] == "4":
            letter = [[0,1,1,5],[0,6,6,1],[3,5,1,1],[3,7,1,2]]
        elif str(number)[excess+1+len(str(int(number)))] == "5":
            letter = [[0,1,6,1],[0,2,1,2],[0,4,6,1],[5,5,1,3],[0,8,6,1]]
        elif str(number)[excess+1+len(str(int(number)))] == "6":
            letter = [[0,1,6,1],[0,2,1,7],[1,4,5,1],[1,8,5,1],[5,5,1,3]]
        elif str(number)[excess+1+len(str(int(number)))] == "7":
            letter = [[0,1,6,1],[5,2,1,7]]
        elif str(number)[excess+1+len(str(int(number)))] == "8":
            letter = [[0,1,6,1],[0,2,1,6],[5,2,1,6],[0,8,6,1],[1,4,4,1]]
        elif str(number)[excess+1+len(str(int(number)))] == "9":
            letter = [[0,1,6,1],[0,2,1,3],[5,2,1,6],[0,8,6,1],[1,4,4,1]]
        elif str(number)[excess+1+len(str(int(number)))] == "0":
            letter = [[0,1,6,1],[0,2,1,6],[5,2,1,6],[0,8,6,1]]
        for i in range (0,len(letter)):
            pygame.draw.rect(win, (0,0,200), (co_ord[0]+(rear+letter[i][0]+8*(excess+len(str(int(number))))+2)*scale,co_ord[1]-5*scale+letter[i][1]*scale,letter[i][2]*scale,letter[i][3]*scale))


def drawleft(number, co_ord, win, scale):
    rear = 0
    for excess in range (0, len(str(number))):
        letter = []
        if str(number)[excess] == "1":
            letter = [[0,1,3,1],[3,2,1,6],[0,8,6,1]]
        elif str(number)[excess] == "2":
            letter = [[0,1,6,1],[5,2,1,3],[0,4,6,1],[0,4,1,4],[0,8,6,1]]
        elif str(number)[excess] == "3":
            letter = [[0,1,5,1],[5,1,1,8],[3,4,2,1],[0,8,5,1]]
        elif str(number)[excess] == "4":
            letter = [[0,1,1,5],[0,6,6,1],[3,5,1,1],[3,7,1,2]]
        elif str(number)[excess] == "5":
            letter = [[0,1,6,1],[0,2,1,2],[0,4,6,1],[5,5,1,3],[0,8,6,1]]
        elif str(number)[excess] == "6":
            letter = [[0,1,6,1],[0,2,1,7],[1,4,5,1],[1,8,5,1],[5,5,1,3]]
        elif str(number)[excess] == "7":
            letter = [[0,1,6,1],[5,2,1,7]]
        elif str(number)[excess] == "8":
            letter = [[0,1,6,1],[0,2,1,6],[5,2,1,6],[0,8,6,1],[1,4,4,1]]
        elif str(number)[excess] == "9":
            letter = [[0,1,6,1],[0,2,1,3],[5,2,1,6],[0,8,6,1],[1,4,4,1]]
        elif str(number)[excess] == "0":
            letter = [[0,1,6,1],[0,2,1,6],[5,2,1,6],[0,8,6,1]]
        for i in range (0,len(letter)):
            pygame.draw.rect(win, (0,0,200), (co_ord[0]+(rear+letter[i][0]+8*excess)*scale,co_ord[1]+letter[i][1]*scale,letter[i][2]*scale,letter[i][3]*scale))

def drawdecleft(number, place, co_ord, win, scale):
    scale = int(scale)
    rear = 0
    if place > len(str(number)) - 1 - len(str(int(number))):
        place = len(str(number)) - 1 - len(str(int(number)))
    for excess in range (0, len(str(int(number)))):
        letter = []
        if str(number)[excess] == "1":
            letter = [[0,1,3,1],[3,2,1,6],[0,8,6,1]]
        elif str(number)[excess] == "2":
            letter = [[0,1,6,1],[5,2,1,3],[0,4,6,1],[0,4,1,4],[0,8,6,1]]
        elif str(number)[excess] == "3":
            letter = [[0,1,5,1],[5,1,1,8],[3,4,2,1],[0,8,5,1]]
        elif str(number)[excess] == "4":
            letter = [[0,1,1,5],[0,6,6,1],[3,5,1,1],[3,7,1,2]]
        elif str(number)[excess] == "5":
            letter = [[0,1,6,1],[0,2,1,2],[0,4,6,1],[5,5,1,3],[0,8,6,1]]
        elif str(number)[excess] == "6":
            letter = [[0,1,6,1],[0,2,1,7],[1,4,5,1],[1,8,5,1],[5,5,1,3]]
        elif str(number)[excess] == "7":
            letter = [[0,1,6,1],[5,2,1,7]]
        elif str(number)[excess] == "8":
            letter = [[0,1,6,1],[0,2,1,6],[5,2,1,6],[0,8,6,1],[1,4,4,1]]
        elif str(number)[excess] == "9":
            letter = [[0,1,6,1],[0,2,1,3],[5,2,1,6],[0,8,6,1],[1,4,4,1]]
        elif str(number)[excess] == "0":
            letter = [[0,1,6,1],[0,2,1,6],[5,2,1,6],[0,8,6,1]]
        for i in range (0,len(letter)):
            pygame.draw.rect(win, (0,0,200), (co_ord[0]+(rear+letter[i][0]+8*excess)*scale,co_ord[1]+letter[i][1]*scale,letter[i][2]*scale,letter[i][3]*scale))
    
    pygame.draw.rect(win, (0,0,200), (co_ord[0]+(rear+8*len(str(int(number))))*scale, co_ord[1]+8*scale, 2*scale, 2*scale))
    
    for excess in range (0, place):
        letter = []
        if str(number)[excess+1+len(str(int(number)))] == "1":
            letter = [[0,1,3,1],[3,2,1,6],[0,8,6,1]]
        elif str(number)[excess+1+len(str(int(number)))] == "2":
            letter = [[0,1,6,1],[5,2,1,3],[0,4,6,1],[0,4,1,4],[0,8,6,1]]
        elif str(number)[excess+1+len(str(int(number)))] == "3":
            letter = [[0,1,5,1],[5,1,1,8],[3,4,2,1],[0,8,5,1]]
        elif str(number)[excess+1+len(str(int(number)))] == "4":
            letter = [[0,1,1,5],[0,6,6,1],[3,5,1,1],[3,7,1,2]]
        elif str(number)[excess+1+len(str(int(number)))] == "5":
            letter = [[0,1,6,1],[0,2,1,2],[0,4,6,1],[5,5,1,3],[0,8,6,1]]
        elif str(number)[excess+1+len(str(int(number)))] == "6":
            letter = [[0,1,6,1],[0,2,1,7],[1,4,5,1],[1,8,5,1],[5,5,1,3]]
        elif str(number)[excess+1+len(str(int(number)))] == "7":
            letter = [[0,1,6,1],[5,2,1,7]]
        elif str(number)[excess+1+len(str(int(number)))] == "8":
            letter = [[0,1,6,1],[0,2,1,6],[5,2,1,6],[0,8,6,1],[1,4,4,1]]
        elif str(number)[excess+1+len(str(int(number)))] == "9":
            letter = [[0,1,6,1],[0,2,1,3],[5,2,1,6],[0,8,6,1],[1,4,4,1]]
        elif str(number)[excess+1+len(str(int(number)))] == "0":
            letter = [[0,1,6,1],[0,2,1,6],[5,2,1,6],[0,8,6,1]]
        for i in range (0,len(letter)):
            pygame.draw.rect(win, (0,0,200), (co_ord[0]+(rear+letter[i][0]+8*(excess+len(str(int(number))))+2)*scale,co_ord[1]+letter[i][1]*scale,letter[i][2]*scale,letter[i][3]*scale))
