import Cirilla_5_2 as ci
import matplotlib.pyplot as plt
import numberlib_v2 as numberlib
import random, pygame, time, sys, gzip, os, webbrowser

def interface(win, option, tite):
    spacer = [40, 160, 280]
    spacery = [120, 240]
    win.fill((255,255,255))
    log = []
    title([200, 20], tite, win)
    for i in range (0, 3):
        button([spacer[i], spacer[i]+80, spacery[0], spacery[0]+60], option[i], (189,32,32),win)
        log.append([spacer[i], spacer[i]+80, spacery[0], spacery[0]+60])
    for i in range (0, 3):
        button([spacer[i], spacer[i]+80, spacery[1], spacery[1]+60], option[i+3], (189,32,32),win)
        log.append([spacer[i], spacer[i]+80, spacery[1], spacery[1]+60])
    local = pygame.mouse.get_pos()
    pressee = pygame.mouse.get_pressed()
    prese = 0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        if event.type == pygame.MOUSEBUTTONUP:
            prese = 1
    buttonp = -1
    for i in range (0, 3):
        if local[0] > log[i][0] and local[0] < log[i][1] and local[1] > log[i][2] and local[1] < log[i][3]:
            if pressee[0] == 1:
                button([spacer[i]+1, spacer[i]+79, spacery[0]+1, spacery[0]+59], option[i], (128,11,11),win)
                log.append([spacer[i], spacer[i]+80, spacery[0], spacery[0]+60])
            else:
                button([spacer[i]+1, spacer[i]+79, spacery[0]+1, spacery[0]+59], option[i], (145,19,19),win)
                log.append([spacer[i], spacer[i]+80, spacery[0], spacery[0]+60])
            if prese == 1:
                buttonp = i
    for i in range (0, 3):
        if local[0] > log[i+3][0] and local[0] < log[i+3][1] and local[1] > log[i+3][2] and local[1] < log[i+3][3]:
            if pressee[0] == 1:
                button([spacer[i]+1, spacer[i]+79, spacery[1]+1, spacery[1]+59], option[i+3], (128,11,11),win)
                log.append([spacer[i], spacer[i]+80, spacery[1], spacery[1]+60])
            else:
                button([spacer[i]+1, spacer[i]+79, spacery[1]+1, spacery[1]+59], option[i+3], (145,19,19),win)
                log.append([spacer[i], spacer[i]+80, spacery[1], spacery[1]+60])
            if prese == 1:
                buttonp = i + 3
    if buttonp == -1:
        pass
    else:
        return option[buttonp]
    pygame.display.update()

def button(coords, written, colour, win):
    font = pygame.font.Font(get_path()+'Roboto.ttf', 12)
    pygame.draw.rect(win, colour, [coords[0], coords[2], coords[1]-coords[0], coords[3]-coords[2]])
    text_width, text_height = font.size(written)
    win.blit(font.render(written, True, (0,0,0)), (coords[0]+int(40-(0.5*text_width)),coords[2]+22))

def title(coords, written, win):
    font = pygame.font.Font(get_path()+'Roboto.ttf', 12)
    text_width, text_height = font.size(written)
    win.blit(font.render(written, True, (0,0,0)), (coords[0]+int(-(0.5*text_width)),coords[1]+22))

def bi_add(legodimensions, recursion, subre, lst):
    if lst == True:
        network = ci.network(legodimensions, LSTM = recursion, alpha = 0.05)
    else:
        network = ci.network(legodimensions, recursion = recursion, alpha = 0.05)

    tot = 0
    go = 0

    win = network.init_display()
    if subre == False:
        if lst == True:
            pygame.display.set_caption('LSTM binary addition ' + str(legodimensions))
        else:
            pygame.display.set_caption('binary addition ' + str(legodimensions))
    else:    
        pygame.display.set_caption('sample of binary addition ' + str(legodimensions))
    set_i_want = 1000
    poppety = [0 for i in range (0, set_i_want)]
    ping = 0

    network.hardlock_display()

    if subre == True:
        network.sample_display()

    while tot != set_i_want:
        go += 1
        if lst == True:
            number1 = random.randint(9999, 99999)
            number2 = random.randint(9999, 99999)
        else:
            number1 = random.randint(0, 99999)
            number2 = random.randint(0, 99999)
        number3 = number1 + number2

        bi_no1 = bin(number1)
        bi_no2 = bin(number2)
        bi_no3 = bin(number3)

        if len(bi_no1) >= len(bi_no2) and len(bi_no1) >= len(bi_no3):
            max_leng = len(bi_no1)
        elif len(bi_no2) >= len(bi_no1) and len(bi_no2) >= len(bi_no3):
            max_leng = len(bi_no2)
        elif len(bi_no3) >= len(bi_no1) and len(bi_no3) >= len(bi_no2):
            max_leng = len(bi_no3)
            
        ext = max_leng - len(str(bi_no1))
        bi_no12 = ""
        for i in range (0, ext):
            bi_no12 += "0"
        for i in range (2,  2+len(str(bi_no1))):
            bi_no12 += str(bi_no1[i:i+1])
        bi_no1 = bi_no12
        ext = max_leng - len(str(bi_no2))
        bi_no22 = ""
        for i in range (0, ext):
            bi_no22 += "0"
        for i in range (2,  2+len(str(bi_no2))):
            bi_no22 += str(bi_no2[i:i+1])
        bi_no2 = bi_no22
        ext = max_leng - len(str(bi_no3))
        bi_no32 = ""
        for i in range (0, ext):
            bi_no32 += "0"
        for i in range (2,  2+len(str(bi_no3))):
            bi_no32 += str(bi_no3[i:i+1])
        bi_no3 = bi_no32

        list1 = [[0,0] for i in range (0, max_leng-2)]
        for intt in range (0, max_leng-2):
            list1[intt] = [int(bi_no1[max_leng-3-intt:max_leng-2-intt]), int(bi_no2[max_leng-3-intt:max_leng-2-intt])]
        list2 = [[0] for i in range (0, max_leng-2)]
        for intt in range (0, max_leng-2):
            list2[intt] = [int(bi_no3[max_leng-3-intt:max_leng-2-intt])]

        #print(number1, number2, number3)
        #print(list1)
        #print(list2)

        cost, ans = network.learn(list1, list2)

        if ans == list2:
            poppety[ping] = 1
        else:
            poppety[ping] = 0
            last_error = ping

        t = 0
        for i in range (0, len(cost)):
            t += cost[i]

        tot = 0
        for i in range (0, set_i_want):
            tot += poppety[i]
        
        network.update_display_cache([t, tot, go-1, ping])

        ping += 1

        if ping >= set_i_want:
            ping = 0

    network.save("2")

    bi_add_test(network)

def bi_add_test(network):
    pygame.quit()
    win = pygame.display.set_mode((400,400))
    pygame.display.set_caption('binary addition test')
    while True:
        number1 = input_number(win, "input number between 0 and 99999:")
        number2 = input_number(win, "input number between 0 and 99999:")
        number3 = number1 + number2

        bi_no1 = bin(number1)
        bi_no2 = bin(number2)
        bi_no3 = bin(number3)

        if len(bi_no1) >= len(bi_no2) and len(bi_no1) >= len(bi_no3):
            max_leng = len(bi_no1)
        elif len(bi_no2) >= len(bi_no1) and len(bi_no2) >= len(bi_no3):
            max_leng = len(bi_no2)
        elif len(bi_no3) >= len(bi_no1) and len(bi_no3) >= len(bi_no2):
            max_leng = len(bi_no3)
            
        ext = max_leng - len(str(bi_no1))
        bi_no12 = ""
        for i in range (0, ext):
            bi_no12 += "0"
        for i in range (2,  2+len(str(bi_no1))):
            bi_no12 += str(bi_no1[i:i+1])
        bi_no1 = bi_no12
        ext = max_leng - len(str(bi_no2))
        bi_no22 = ""
        for i in range (0, ext):
            bi_no22 += "0"
        for i in range (2,  2+len(str(bi_no2))):
            bi_no22 += str(bi_no2[i:i+1])
        bi_no2 = bi_no22
        ext = max_leng - len(str(bi_no3))
        bi_no32 = ""
        for i in range (0, ext):
            bi_no32 += "0"
        for i in range (2,  2+len(str(bi_no3))):
            bi_no32 += str(bi_no3[i:i+1])
        bi_no3 = bi_no32

        list1 = [[0,0] for i in range (0, max_leng-2)]
        for intt in range (0, max_leng-2):
            list1[intt] = [int(bi_no1[max_leng-3-intt:max_leng-2-intt]), int(bi_no2[max_leng-3-intt:max_leng-2-intt])]
        list2 = [[0] for i in range (0, max_leng-2)]
        for intt in range (0, max_leng-2):
            list2[intt] = [int(bi_no3[max_leng-3-intt:max_leng-2-intt])]

        ans = network.est(list1)

        asnd = ""

        for i in range (0, len(ans)):
            asnd += str(ans[ext-i-1][0])

        int_outp = 0
        for i in range (len(asnd), -1, -1):
            if asnd[i:i+1] == "1":
                int_outp += 2 ** (len(asnd)-i-1)

        bi111 = ""

        for i in range (0, len(bi_no1)):
            bi111 += str(bi_no1[i][0])

        bi222 = ""

        for i in range (0, len(bi_no2)):
            bi222 += str(bi_no2[i][0])

        bi333 = ""

        for i in range (0, len(bi_no3)):
            bi333 += str(bi_no3[i][0])

        num_d = ["number 1: "+str(number1), "number 2: "+str(number2), "output: "+str(int_outp), "correct: "+str(number3)]
        bin_d = ["number 1: "+bi111, "number 2: "+bi222, "output: "+asnd, "correct: "+bi333]
        
        output_data(win, num_d, bin_d)

def output_data(win, number_d, bi_d):
    pygame.font.init()
    font = pygame.font.Font('Roboto.ttf', 12)
    exist = True
    pygame.display.set_caption('results')

    while exist == True:
        win.fill((255,255,255))
        win.blit(font.render(number_d[0], True, (0,0,0)), (25,25))
        win.blit(font.render(number_d[1], True, (0,0,0)), (25,125))
        win.blit(font.render(number_d[2], True, (0,0,0)), (25,225))
        win.blit(font.render(number_d[3], True, (0,0,0)), (25,325))
        win.blit(font.render(bi_d[0], True, (0,0,0)), (150,25))
        win.blit(font.render(bi_d[1], True, (0,0,0)), (150,125))
        win.blit(font.render(bi_d[2], True, (0,0,0)), (150,225))
        win.blit(font.render(bi_d[3], True, (0,0,0)), (150,325))
        if number_d[2][8:] == number_d[3][9:]:
            win.blit(font.render("Correct", True, (0,250,0)), (150,350))
        else:
            win.blit(font.render("Wrong", True, (250,0,0)), (150,350))
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    exist = False
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
        pygame.display.flip()

def input_number(win, tex):
    pygame.font.init()
    font = pygame.font.Font('Roboto.ttf', 12)
    input_str = ""
    exist = True
    pygame.display.set_caption('input numbers')
    
    while exist == True:
        win.fill((255,255,255))
        text_width, text_height = font.size(tex)
        win.blit(font.render(tex, True, (0,0,0)), (int(200-(0.5*text_width)),50))
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0:
                    input_str += "0"
                if event.key == pygame.K_1:
                    input_str += "1"
                if event.key == pygame.K_2:
                    input_str += "2"
                if event.key == pygame.K_3:
                    input_str += "3"
                if event.key == pygame.K_4:
                    input_str += "4"
                if event.key == pygame.K_5:
                    input_str += "5"
                if event.key == pygame.K_6:
                    input_str += "6"
                if event.key == pygame.K_7:
                    input_str += "7"
                if event.key == pygame.K_8:
                    input_str += "8"
                if event.key == pygame.K_9:
                    input_str += "9"
                if event.key == pygame.K_BACKSPACE:
                    input_str = input_str[0:len(input_str)-1]
                if event.key == pygame.K_RETURN:
                    exist = False
        if len(input_str) > 0:
            numberlib.draw(int(input_str), [200,250], win, 2)
        pygame.display.flip()
    return int(input_str)
        

def mnist(arce, sube, lensws, epoche, alph):
    network = ci.network(arce, alpha = alph, bias = True)#0.01625
    network.sample_display()
    
    start_time = time.time()
    last_time = start_time
    last_list = 0

    file = open(get_path()+"settings.txt", 'r+')
    email_permission = int(file.readline().rstrip())
    target_adress = str(file.readline().rstrip())

    if email_permission == 1:
        network.init_mail()
        network.mail_send(target_adress, "network running", "your " + str(arce) + " network has started training")

    f = gzip.open(get_path()+'train-images-idx3-ubyte.gz','r')
    f.read(16)
    f1 = gzip.open(get_path()+'train-labels-idx1-ubyte.gz','r')
    f1.read(8)

    #network.init_display()
    #network.hardlock_display()

    maxi = lensws

    win = network.init_display()
    network.hardlock_display()
    pygame.display.set_caption('training from the MNIST dataset ' + str(arce))

    grid = [0 for j in range (0, 784)]
    grids = [[0 for j in range (0, 784)] for k in range (0, maxi+100)]
    results = [[0 for j in range (0, 10)] for i in range (0, maxi+100)]
    poppety = [0 for i in range (0, epoche)]

    for i in range (0, maxi+100):
        if i%100 == 0:
            network.update_display_cache([i])
            blank = network.generate_f()
            network.update_display(blank, 0, 0)
        for k in range (0, 784):
            grids[i][k] = (ord(f.read(1))/255)
            if grids[i][k] > 1:
                print("hhdhd")
            elif grids[i][k] < 0:
                print("dsadsa")
        results[i][ord(f1.read(1))] = 1

    list6 = network.order(int(maxi/epoche))
    etia = 0
    accurat = [0 for i in range (0, int(maxi/epoche))]

    for i in range (0, int(maxi/epoche)):
        list1 = network.order(epoche)
        loop_start = time.time()
        for j in range (0, epoche):
            grid = grids[list6[i]*epoche+list1[j]]

            cost, ans = network.learn(grid, results[list6[i]*epoche+list1[j]])

            if ans == results[list6[i]*epoche+list1[j]]:
                poppety[j] = 1
            else:
                poppety[j] = 0

            tot = 0
            for h in range (0, epoche):
                tot += poppety[h]
            network.update_display_cache([cost, tot, j, i*epoche+j])
            if (time.time() - last_time) > 1 and email_permission == 1:
                network.mail_send(target_adress, "update", "in the last test the netork got " + str(accurat[i-1]) + " out of 100")
                last_time = time.time()
        accurat[i] = mnist_test(results, grids, maxi, network)
    if email_permission == 1:
        network.mail_send(target_adress, "complete", "in the last test the netork got " + str(int(maxi/epoche)-1) + " out of 100")
    network.save("1")
    plt.plot(accurat)
    plt.show()

def mnist_test(result, number, maxi, network):
    acc = 0
    for i in range (0, 100):
        rerre = network.est(number[maxi+i])
        if rerre == result[maxi+i]:
            acc += 1
    return acc

def mnist_t(network):
    network.sample_display()
    f = gzip.open(get_path()+'t10k-images-idx3-ubyte.gz','r')
    f.read(16)
    f1 = gzip.open(get_path()+'t10k-labels-idx1-ubyte.gz','r')
    f1.read(8)

    maxi = 1000

    win = network.init_display()
    network.hardlock_display()
    pygame.display.set_caption('testing from the MNIST dataset')

    grid = [0 for j in range (0, 784)]
    grids = [[0 for j in range (0, 784)] for k in range (0, maxi)]
    results = [[0 for j in range (0, 10)] for i in range (0, maxi)]
    total = [0 for j in range (0, maxi)]

    for i in range (0, maxi):
        if i%100 == 0:
            network.update_display_cache([i])
            blank = network.generate_f()
            network.update_display(blank, 0, 0)
        for k in range (0, 784):
            grids[i][k] = (ord(f.read(1))/255)
            if grids[i][k] > 1:
                print("hhdhd")
            elif grids[i][k] < 0:
                print("dsadsa")
        results[i][ord(f1.read(1))] = 1
    acc = 0
    for i in range (0, maxi):
        rerre = network.est(grids[i])
        if rerre == results[i]:
            acc += 1
        total[i] = acc
    plt.plot(total)
    plt.show()

def loadn(fil):
    network = ci.network([1,1,1])
    if True:
        if fil == "con":
            network.load("1")
            mnist_t(network)
        elif fil == "rec":
            network.load("2")
            bi_add_test(network)
        else:
            print("error")
    #except:
     #   print("no file")

class bird():

    def __init__(self, screen, no):
        self.number = no
        self.gravity = 0.4
        self.bird_movement = 0
        self.game_active = True
        self.score = 0
        self.high_score = 0
        self.screen = screen

        bg_surface = pygame.image.load(get_path()+'assets/background-day.png').convert()
        self.bg_surface = pygame.transform.scale2x(bg_surface)

        floor_surface = pygame.image.load(get_path()+'assets/base.png').convert()
        self.floor_surface = pygame.transform.scale2x(floor_surface)
        self.floor_x_pos = 0

        bird_downflap = pygame.transform.scale2x(pygame.image.load(get_path()+'assets/bluebird-downflap.png').convert_alpha())
        bird_midflap = pygame.transform.scale2x(pygame.image.load(get_path()+'assets/bluebird-midflap.png').convert_alpha())
        bird_upflap = pygame.transform.scale2x(pygame.image.load(get_path()+'assets/bluebird-upflap.png').convert_alpha())
        self.bird_frames = [bird_downflap,bird_midflap,bird_upflap]
        self.bird_index = 0
        self.bird_surface = self.bird_frames[self.bird_index]
        self.bird_rect = self.bird_surface.get_rect(center = (100,512))

        self.BIRDFLAP = pygame.USEREVENT + 1
        pygame.time.set_timer(self.BIRDFLAP,200)
        self.life = True

        # bird_surface = pygame.image.load('assets/bluebird-midflap.png').convert_alpha()
        # bird_surface = pygame.transform.scale2x(bird_surface)
        # bird_rect = bird_surface.get_rect(center = (100,512))

    def do(self, pipes, neat, pipe_list, screen, maxi, pipe_height, pipe_surface):

        if self.life:
            self.addi = 0
            if len(pipes)%2 == 1:
                ii = 1
            else:
                ii = 0
            distance = pipes[ii][0]-self.bird_rect[0] + pipes[ii][2]
            while distance < 0:
                ii += 2
                distance = pipes[ii][0]-self.bird_rect[0]
            top = self.bird_rect[1]-pipes[ii+1][1]-pipes[ii+1][3]
            bottom = pipes[ii][1]-self.bird_rect[1]
            if distance < 100 and len(pipes) <= ii+2:
                pipe_list.extend(create_pipe(pipe_height, pipe_surface))
            feed_info = [distance/100, top/100, bottom/100, (-1*self.bird_movement)/10]
            action = neat.predict(feed_info, self.number)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == self.BIRDFLAP:
                    if self.bird_index < 2:
                        self.bird_index += 1
                    else:
                        self.bird_index = 0

                    bird_surface,bird_rect = self.bird_animation()

            if action[0] >= 0.5:
                self.bird_movement = 0
                self.bird_movement -= 12
        
                    # Bird
            self.bird_movement += self.gravity
            rotated_bird = self.rotate_bird(self.bird_surface)
            self.bird_rect.centery += self.bird_movement
            self.screen.blit(rotated_bird,self.bird_rect)
            self.life, self.addi = self.check_collision(pipe_list)
                    
            self.score += 1

            if self.life == False:
                if self.addi == 0 and self.score ** 2 >= maxi:
                    self.breed = self.score ** 2
                else:
                    self.breed = 0
            
            return True, 0.0
        else:
            return False, self.breed


    def check_collision(self, pipes):
        for pipe in pipes:
            if self.bird_rect.colliderect(pipe):
                return False, 0

        if self.bird_rect.top <= -100 or self.bird_rect.bottom >= 900:
            return False, 1

        return True, 2

    def rotate_bird(self, bird):
        new_bird = pygame.transform.rotozoom(bird,-self.bird_movement * 3,1)
        return new_bird

    def bird_animation(self):
        new_bird = self.bird_frames[self.bird_index]
        new_bird_rect = new_bird.get_rect(center = (100,self.bird_rect.centery))
        return new_bird,new_bird_rect

    def score_display(self, game_state):
        if game_state == 'main_game':
            score_surface = game_font.render(str(int(score)),True,(255,255,255))
            score_rect = score_surface.get_rect(center = (288,100))
            screen.blit(score_surface,score_rect)
        if game_state == 'game_over':
            score_surface = game_font.render(f'Score: {int(score)}' ,True,(255,255,255))
            score_rect = score_surface.get_rect(center = (288,100))
            screen.blit(score_surface,score_rect)

            high_score_surface = game_font.render(f'High score: {int(high_score)}',True,(255,255,255))
            high_score_rect = high_score_surface.get_rect(center = (288,850))
            screen.blit(high_score_surface,high_score_rect)

    def update_score(self, score, high_score):
        if score > high_score:
            high_score = score
        return high_score

    def reset(self):
        self.life = True
        self.bird_rect.center = (100,512)
        self.bird_movement = 0
        self.score = 0

def create_pipe(pipe_height, pipe_surface):
    random_pipe_pos = random.choice(pipe_height)
    bottom_pipe = pipe_surface.get_rect(midtop = (1000,random_pipe_pos))
    top_pipe = pipe_surface.get_rect(midbottom = (1000,random_pipe_pos - 300))
    return bottom_pipe,top_pipe

def move_pipes(pipes):
    for pipe in pipes:
        pipe.centerx -= 5
    return pipes

def draw_pipes(pipes, screen, pipe_surface):
    for pipe in pipes:
        if pipe.bottom >= 1024:
            screen.blit(pipe_surface,pipe)
        else:
            flip_pipe = pygame.transform.flip(pipe_surface,False,True)
            screen.blit(flip_pipe,pipe)

def remove_pipes(pipes):
    for pipe in pipes:
        if pipe.centerx < -600:
            pipes.remove(pipe)
    return pipes
    
def draw_floor(screen, floor_surface, floor_x_pos):
    screen.blit(floor_surface,(floor_x_pos,900))
    screen.blit(floor_surface,(floor_x_pos + 576,900))

def run_flappy(num_of_bird):
    pygame.init()
    screen = pygame.display.set_mode((576,1024))
    clock = pygame.time.Clock()
    game_font = pygame.font.Font(get_path()+'04B_19.ttf',40)
    pipe_surface = pygame.image.load(get_path()+'assets/pipe-green.png')
    pipe_surface = pygame.transform.scale2x(pipe_surface)
    pipe_list = []
    SPAWNPIPE = pygame.USEREVENT
    pygame.time.set_timer(SPAWNPIPE,1200)
    pipe_height = [400,600,800]
    floor_surface = pygame.image.load(get_path()+'assets/base.png').convert()
    floor_surface = pygame.transform.scale2x(floor_surface)
    floor_x_pos = 0
    neat = ci.neat([4,1], num_of_bird, prob = 0.05)
    # Game Variables
    bg_surface = pygame.image.load(get_path()+'assets/background-day.png').convert()
    bg_surface = pygame.transform.scale2x(bg_surface)

    birds = [bird(screen, i) for i in range (0, num_of_bird)]
    maxi = 0

    if __name__ == "__main__":
        while True:
            screen.blit(bg_surface,(0,0))
            if len(pipe_list) > 0:
                pass
            else:
                pipe_list.extend(create_pipe(pipe_height, pipe_surface))
            life = [True for i in range (0, len(birds))]
            scores = [0 for i in range (0, len(birds))]
            for i in range (0, len(birds)):
                life[i], scores[i] = birds[i].do(pipe_list, neat, pipe_list, screen, maxi, pipe_height, pipe_surface)
            if life == [False for i in range (0, len(birds))]:
                neat.bi_generation(scores)
                for i in range (0, len(birds)):
                    birds[i].reset()
                pipe_list.clear()
                for i in range (0, len(scores)):
                    if scores[i] > maxi:
                        maxi = scores[i]
            else:
                pipe_list = move_pipes(pipe_list)
                pipe_list = remove_pipes(pipe_list)
                draw_pipes(pipe_list, screen, pipe_surface)
                draw_floor(screen, floor_surface, floor_x_pos)

                pygame.display.update()
                clock.tick(120)

def settings(win, network):
    file = open(get_path()+"settings.txt", 'r+')
    exist = True
    emi = int(file.readline().rstrip())
    target_email = str(file.readline().rstrip())
    pygame.display.set_caption('email settings')
    if emi == 1:
        emai = True
    elif emi == 0:
        emai = False
    else:
        print("email catch")

    while exist == True:
        prese = 0
        win.fill((255,255,255))
        button([60,140,160,220], "rewrite", (240,0,0), win)
        if emai == True:
            button([260,340,160,220], "turn off", (240,0,0), win)
        else:
            button([260,340,160,220], "turn on", (240,0,0), win)
        button([60,140,230,290], "reset target", (240,0,0), win)
        button([260,340,230,290], "back", (240,0,0), win)
        title([200,20], "email settings:", win)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.MOUSEBUTTONUP:
                prese = 1
                
        pos = pygame.mouse.get_pos()
        apple_p = pygame.mouse.get_pressed()
        if pos[0] > 60 and pos[0] < 140 and pos[1] > 160 and pos[1] < 220:
            if apple_p[0] == 1:
                button([61,139,161,219], "rewrite", (128,11,11), win)
            else:
                button([61,139,161,219], "rewrite", (145,19,19), win)
            if prese == 1:
                os.remove(get_path()+"email.txt")
                network.init_mail()
                win = pygame.display.set_mode((400,360))
                pygame.font.init()
        elif pos[0] > 260 and pos[0] < 340 and pos[1] > 160 and pos[1] < 220:
            if apple_p[0] == 1:
                if emai == True:
                    button([261,339,161,219], "turn off", (128,11,11), win)
                else:
                    button([261,339,161,219], "turn on", (128,11,11), win)
            else:
                if emai == True:
                    button([261,339,161,219], "turn off", (145,19,19), win)
                else:
                    button([261,339,161,219], "turn on", (145,19,19), win)
            if prese == 1:
                if emai == True:
                    emai = False
                    emi = 0
                else:
                    emai = True
                    emi = 1
        elif pos[0] > 60 and pos[0] < 140 and pos[1] > 230 and pos[1] < 290:
            if apple_p[0] == 1:
                button([61,139,231,289], "reset target", (128,11,11), win)
            else:
                button([61,139,231,289], "reset target", (145,19,19), win)
            if prese == 1:
                win = pygame.display.set_mode((200,200))
                emaiss = network.input_se("input email adress up to @:", win)
                emaiss += "@"
                win = pygame.display.set_mode((200,200))
                emaiss += network.input_se("input email adress after @:", win)
                target_email = emaiss
                win = pygame.display.set_mode((400,360))
                pygame.font.init()
        elif pos[0] > 260 and pos[0] < 340 and pos[1] > 230 and pos[1] < 290:
            if apple_p[0] == 1:
                button([261,339,231,289], "back", (128,11,11), win)
            else:
                button([261,339,231,289], "back", (145,19,19), win)
            if prese == 1:
                exist = False
        
        pygame.display.flip()
    file.close()
    os.remove(get_path()+"settings.txt")
    file = open(get_path()+"settings.txt", "w")
    file.write(str(emi))
    file.write("\n")
    file.write(target_email)
    file.close()
    return win


def get_path():
    path = os.path.realpath(__file__)
    ex = True
    while ex == True:
        if path[len(path)-1:len(path)] != '\\':
            path = path[0:len(path)-1]
        else:
            ex = False
    return path

if __name__ == "__main__":
    network = ci.network([1,1,1])
    network.init_mail()
    try:
        file = open(get_path()+"settings.txt", 'r+')
        file.close()
    except:
        file = open(get_path()+"settings.txt", 'w')
        file.write(str(1))
        file.write("\n")
        filili = open(get_path()+"email.txt")
        file.write(filili.readline().rstrip())
        file.close()
        filili.close()

    function = False
    base = ""
    out = False
    pygame.display.init()
    pygame.font.init()
    win = pygame.display.set_mode((400,360))
    pygame.display.set_caption('UI')
    responce = 0
    dave = 0
    kevin = 0
    while out == False:
        option = ["recursion","convolution","NEAT","documentation","exit","email"]
        responce = interface(win, option, "select a network type:")
        while responce == "recursion":
            time.sleep(0.1)
            option = ["[2, 10, 1]", "LSTM[2, 10, 1]", "[2, 50[10], 1]", "[2, 5, 1]", "load", "back"]
            responce = interface(win, option, "choose a recursion architecture:")
            if responce == None:
                responce = "recursion"
            if responce != "back" and responce != "recursion":
                base = "recursion"
                function = True
                break
        while responce == "convolution":
            time.sleep(0.1)
            option = ["[784, 32, 10]", "C[784, 10, 10]","C[784,10,10,10]", "[784,32,32,10]", "load", "back"]
            responce = interface(win, option, "choose a convolution/FF architecture:")
            if responce == None:
                responce = "convolution"
            if responce != "back" and responce != "convolution":
                base = "convolution"
                function = True
                break
        while responce == "NEAT":
            time.sleep(0.1)
            option = ["10", "50","100","","","back"]
            responce = interface(win, option, "choose a number of NEAT items:")
            if responce == None:
                responce = "NEAT"
            if responce != "back" and responce != "NEAT":
                base = "NEAT"
                function = True
                break
        if responce == "documentation":
            print("documentation")
            os.system('start doc.txt')
            out = True
        if responce == "email":
            win = settings(win, network)
            pygame.display.set_caption('UI')
            responce = ""
        if responce == "exit":
            out = True
        if function == True:
            if base == "recursion":
                #["[2, 10, 1]", "LSTM[2, 10, 1]", "[2, 50[10], 1]", "[2, 3, 1]", "load", "back"]
                if responce == "[2, 10, 1]":
                    bi_add([2, 10, 1], [1], False, False)
                elif responce == "LSTM[2, 10, 1]":
                    bi_add([2, 10, 1], [1], False, True)
                elif responce == "[2, 50[10], 1]":
                    bi_add([2, 50, 1], [1], True, False)
                elif responce == "[2, 5, 1]":
                    bi_add([2, 5, 1], [1], False, False)
                elif responce == "load":
                    loadn("rec")
            if base == "convolution":
                #["[784, 32, 10]", "C[784, 10, 10]","C[784,10,10,10]", "[784,32,32,10]", "load", "back"]
                if responce == "[784, 32, 10]":
                    mnist([784, 32, 10], True, 5000, 10, 0.01)
                elif responce == "C[784,10,10,10]":
                    mnist([[28,28],[10,10,3],[10,10,1],[10]], True, 20000, 250, 0.01)
                elif responce == "[784,32,32,10]":
                    mnist([784,32,32,10], True, 30000, 100, 0.01)
                elif responce == "C[784, 10, 10]":
                    mnist([[28,28], [10,10,3], [10]], True, 20000, 250, 0.015)
                elif responce == "load":
                    loadn("con")
            if base == "NEAT":
                #["10","100","500","","","back"]
                if responce == "10":
                    run_flappy(10)
                elif responce == "100":
                    run_flappy(100)
                elif responce == "50":
                    run_flappy(50)
            function = False
            win = pygame.display.set_mode((400,360))
    pygame.quit()
