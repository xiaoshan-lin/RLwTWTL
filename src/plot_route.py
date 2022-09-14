
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.colors import to_rgba
import numpy as np


FILE = '../output/test_policy_trajectory_log.txt'
LINE = 10
DIMS = (6,6)
ENV_INFO = {
    'Pickup':{'loc':'r28', 'color':'xkcd:light yellow'},
    'Delivery':{'loc':'r1', 'color':'cornflowerblue'},
    'Region 1':{'loc':'r9', 'color':'tab:green'},
    'Region 2':{'loc':'r32', 'color':'tab:green'}
    # 'R2':{'loc':'r16', 'color':'green'}
}

AGENT_WIDTH = 0.7


class BasePlot(object):


    def __init__(self, dims, env_info, plots = 1, direction = 'rows'):

        if direction == 'rows':
            ncols = 1
            nrows = plots
        elif direction == 'columns':
            ncols = plots
            nrows = 1
        else:
            raise Exception('Error: Valid plot directions are "rows" or "columns"')

        self.width = dims[0]
        self.height = dims[1]
        scale = 2
        
        # set up plot
        self.fig, self.axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[self.width*scale*ncols, self.height*scale*nrows], facecolor='xkcd:charcoal grey')

        if type(self.axs) != np.ndarray:
            self.axs = np.array([self.axs])

        self.axs = np.ravel(self.axs)

        self.ax = self.axs[0]

        for a in self.axs:
            a.axis([0,self.width,0,self.height])
            a.grid(True)
            a.grid(linewidth=2)
            a.set_xticks(list(range(0,self.width)))
            a.set_yticks(list(range(0,self.height)))
            # get rid of labels
            a.set_xticklabels([])
            a.set_yticklabels([])
            # get rid of ticks
            for tic in a.xaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
            for tic in a.yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False

            a.invert_yaxis()

            self._draw_env(env_info)

    def _draw_env(self,env_info):
        for l,d in list(env_info.items()):
            r = d['loc'].replace('r', '')
            r = int(r)
            if l == 'Pickup':
                self.pickup_loc = r
            elif l == 'Delivery':
                self.delivery_loc = r
            x, y = self._r2c(r)
            c = d['color']
            for ax in self.axs:
                self._draw_square(x,y,c, ax, label=l)

    def _draw_square(self,x,y,color, ax, label = None):
        x1 = x-0.5
        x2 = x+0.5
        y1 = y-0.5
        y2 = y+0.5
        ax.fill([x1, x2, x2, x1], [y2, y2, y1, y1], color, label = label)

    def _draw_pic(self, x, y, ax, file, color=None, zorder=4, width=AGENT_WIDTH):
        bbox_xy = self._xy_to_bbox_xy((x,y), width)
        bbox = Bbox(bbox_xy)
        bbox_tf = TransformedBbox(bbox, ax.transData)
        bbox_image = BboxImage(bbox_tf,
                       norm=None,
                       origin=None)
        drone_img = mpimg.imread(file)
        if color != None:
            rgba = to_rgba(color)
            not_invis = drone_img[:,:,3] > 0
            drone_img[not_invis] = rgba
        bbox_image.set_data(drone_img)
        bbox_image.set(zorder=zorder)
        ax.add_artist(bbox_image)
        return bbox_image

    def _r2c(self, r):
        y, x = np.unravel_index(r, (self.height, self.width)) # pylint: disable=unbalanced-tuple-unpacking
        x += 0.5
        y += 0.5
        return x, y

    def _xy_to_bbox_xy(self, xy, width=AGENT_WIDTH):
        x0 = xy[0] - width/2
        y0 = xy[1] - width/2
        x1 = xy[0] + width/2
        y1 = xy[1] + width/2
        return np.array([[x0,y0],[x1,y1]])


    def show(self):
        plt.show()


class ArrowPlot(BasePlot):

    TIME_XY = (0.5, 1.9)
    TIME_SZ = 15

    def __init__(self, dims, env_info, path, break_idx = []):
        num_plots = len(break_idx) + 1
        super(ArrowPlot, self).__init__(dims, env_info, plots=num_plots)
        break_idx.insert(0, 0)
        break_idx.append(len(path)-1)
        self.break_idx = break_idx

        # make a path for the package picture
        pack_path = [self.pickup_loc]
        pack_state = 0
        for p1,p2 in zip(path[:-1], path[1:]):
            if pack_state == 0:
                pack_path.append(self.pickup_loc)
                if p1 == p2 == self.pickup_loc:
                    pack_state = 1
            elif pack_state == 1:
                pack_path.append(p2)
                if p1 == p2 == self.delivery_loc:
                    pack_state = 2
            elif pack_state == 2:
                pack_path.append(self.delivery_loc)

        self.paths = []
        self.pack_paths = []
        for a,b in zip(break_idx[:-1], break_idx[1:]):
            self.paths.append(path[a:b+1])
            self.pack_paths.append(pack_path[a:b+1])

        self.path_lens = [len(p) for p in self.paths]

    def get_path_lens(self):
        return self.path_lens

    def draw_path(self, arrow_seq=None, color='black', scale=5):

        # build broken sequences of arrows
        arrow_seqs = []
        for a,b in zip(self.break_idx[:-1], self.break_idx[1:]):
            seq = arrow_seq[a:b+1]
            if seq[0] == 'before':
                seq[0] = 'none'
            if seq[-1] == 'after':
                seq[-1] = 'none'
            arrow_seqs.append(seq)

        for i,p in enumerate(self.paths):
            xdata, ydata = self._path_to_xy(p)
            packx, packy = self._path_to_xy(self.pack_paths[i])
            self._draw_line(xdata, ydata, ax=self.axs[i], color=color, scale=scale)
            self._draw_arrows(xdata, ydata, ax=self.axs[i], color=color, arrow_seq=arrow_seqs[i], scale=scale)
            # self._draw_drone(xdata[0], ydata[0], ax=self.axs[i], color='tab:blue')
            self._draw_agent(xdata[-1], ydata[-1], ax=self.axs[i])
            self._draw_package(packx[-1], packy[-1], ax=self.axs[i])
            self._draw_time(i, self.axs[i])

    def _draw_time(self, i, ax):
        time0 = sum(self.path_lens[:i]) - i
        time1 = time0 + self.path_lens[i] - 1
        if time0 == time1:
            txt = 't = {}'.format(time0)
        else:
            txt = 't = {}:{}'.format(time0, time1)
        x,y = self.TIME_XY
        ax.text(x, y, txt, size=self.TIME_SZ, ha='center')

    def _draw_line(self, xdata, ydata, ax, scale=5, color='black'):
        line = Line2D(xdata, ydata, linewidth=scale, color=color)
        ax.add_artist(line)

    def _draw_arrows(self, x, y, ax, arrow_seq=None, color='black', scale=5):
        # Used ideas from multiple answers of https://stackoverflow.com/questions/8247973/how-do-i-specify-an-arrow-like-linestyle-in-matplotlib
        BEFORE_LEN = 0.8
        AFTER_LEN = 0.4
        ARROW_SCALE = 6
        LOOP_DIA = 0.5
        LOOP_ROT_ANG = 0
        LOOP_END_ANG = 300

        if arrow_seq == None:
            raise Exception('Smart arrow making not implemented. Use arrow_seq.')
        elif len(arrow_seq) != len(x):
            raise Exception('Arrow sequence must define arrow type for each movement')
        elif arrow_seq[0] == 'before' or arrow_seq[-1] == 'after':
            raise Exception('Cannot have arrow before first or after last')

        theta = np.arctan2(-(y[1:] - y[:-1]), x[1:] - x[:-1])
        path = np.array([x,y]).T
        dist = np.sum((path[1:] - path[:-1]) ** 2, axis=1) ** .5

        ax0 = []
        ax1 = []
        ay0 = []
        ay1 = []
        for i, arr_type in enumerate(arrow_seq):
            if arr_type == 'none':
                pass
            elif arr_type == 'before':
                ax0.append(x[i-1])
                ax1.append(x[i-1] + dist[i-1] * np.cos(theta[i-1]) * BEFORE_LEN)
                ay0.append(y[i-1])
                ay1.append(y[i-1] + dist[i-1] * -np.sin(theta[i-1]) * BEFORE_LEN)
            elif arr_type == 'after':
                ax0.append(x[i])
                ax1.append(x[i] + dist[i] * np.cos(theta[i]) * AFTER_LEN)
                ay0.append(y[i])
                ay1.append(y[i] + dist[i] * -np.sin(theta[i]) * AFTER_LEN)
            elif arr_type == 'loop':
                # https://stackoverflow.com/a/38208040
                # Line
                arc = mpatches.Arc([x[i], y[i]], LOOP_DIA, LOOP_DIA, angle=LOOP_ROT_ANG, theta1=0, theta2=LOOP_END_ANG, capstyle='round', lw=scale, color=color, zorder=1.5)
                # arrow
                head_ang = LOOP_ROT_ANG+LOOP_END_ANG
                endx=x[i]+(LOOP_DIA/2)*np.cos(np.radians(head_ang)) #Do trig to determine end position
                endy=y[i]+(LOOP_DIA/2)*np.sin(np.radians(head_ang))
                head = mpatches.RegularPolygon((endx,endy), 3, LOOP_DIA/9, np.radians(head_ang), color=color, zorder=1.5)
                ax.add_patch(arc)
                ax.add_patch(head)
            else:
                raise Exception('Unrecognized arrow type/location: ' + arr_type)
                
        for x1, y1, x2, y2 in zip(ax0, ay0, ax1, ay1):
            ax.annotate('', xytext=(x1, y1), xycoords='data',
                xy=(x2, y2), textcoords='data',
                arrowprops=dict(arrowstyle='-|>', mutation_scale=scale*ARROW_SCALE, color='black', zorder=1.5))#, frac=1., ec=c, fc=c))

    def _draw_agent(self, x, y, ax, color=None):
        file = '../lib/drone.png'
        self._draw_pic(x, y, ax, file, color=color)

    def _draw_package(self, x, y, ax):
        file = '../lib/package.png'
        self._draw_pic(x, y, ax, file, zorder=3.9, width = 0.4)

    def _path_to_xy(self, path):
        coords = [self._r2c(r) for r in path]
        xdata = np.array([c[0] for c in coords])
        ydata = np.array([c[1] for c in coords])
        return xdata, ydata


class AnimPlot(BasePlot):

    AGENT_WIDTH = 0.7
    AGENT_COLOR = 'blue'
    TIME_XY = (0.1, 1.9)
    TIME_SZ = 15

    def __init__(self, dims, env_info, path, num_steps = 30, time_step = 0.7, episode=None):
        super(AnimPlot, self).__init__(dims, env_info)
        self.path = path
        self.num_steps = num_steps
        self.time_step = time_step

        package_r_path = package_path_gen(self.path, self.pickup_loc, self.delivery_loc)
        self.agent_path = self.gen_smooth_path(path, num_steps)
        self.package_path = self.gen_smooth_path(package_r_path, num_steps)

        # Agent image
        x,y = self.agent_path[0]
        drone_file = '../lib/drone.png'
        self.agent_bbox_image = self._draw_pic(x, y, self.ax, drone_file)

        # Package image
        x,y = self.package_path[0]
        package_file = '../lib/package.png'
        self.pack_bbox_image = self._draw_pic(x, y, self.ax, package_file, zorder=3.9, width=0.4)

        # time
        x,y = self.TIME_XY
        self.time_text = self.ax.text(x, y, 't = 0', size = self.TIME_SZ)

        # episode
        if episode != None:
            x,y = self.TIME_XY
            y -= 0.3
            self.ax.text(x,y, 'Episode {}'.format(episode), size=self.TIME_SZ-5)

        # # init_coords = self._r2c(self.path[0])
        # agent_bbox_xy = self._xy_to_bbox_xy(agent_init_xy)
        # agent_bbox = Bbox(agent_bbox_xy)
        # # self.bbox = Bbox([[2,2],[3,3]])
        # agent_bbox_tf = TransformedBbox(agent_bbox, self.ax.transData)
        # self.agent_bbox_image = BboxImage(agent_bbox_tf,
        #                cmap=plt.get_cmap('winter'),
        #                norm=None,
        #                origin=None)
        # drone_img = mpimg.imread('../lib/drone.png')
        # self.agent_bbox_image.set_data(drone_img)
        # self.ax.add_artist(self.agent_bbox_image)
        # plt.show()

        # self.agent_patch = mpatches.Circle(init_coords, self.AGENT_WIDTH, color=self.AGENT_COLOR)

    def anim_update(self,t):
        p_bbox_xy = self._xy_to_bbox_xy(self.package_path[t], width = 0.4)
        self.pack_bbox_image.bbox._bbox.update_from_data_xy(p_bbox_xy)
        a_bbox_xy = self._xy_to_bbox_xy(self.agent_path[t])
        self.agent_bbox_image.bbox._bbox.update_from_data_xy(a_bbox_xy)
        time = t // self.num_steps
        self.time_text.set_text('t = {}'.format(time))
        return [self.time_text, self.pack_bbox_image, self.agent_bbox_image]

    def gen_smooth_path(self, r_path, numSteps):
        path = self._r_path_to_xy(r_path)
        self.init = path[0]
        smooth_path = [list(zip(np.linspace(c[0],c2[0],numSteps), np.linspace(c[1],c2[1],numSteps))) for c,c2 in zip(path[:-1],path[1:])]
        # flatten
        smooth_path = [item for sub in smooth_path for item in sub] # Maybe someday Python will allow unpacking in list comprehensions...
        return smooth_path
        
    def start(self, show=True):
        interval = int(float(self.time_step) / (self.num_steps) * 1000.0)
        # interval = 500
        sim_frames = len(self.agent_path)
        self.ani = FuncAnimation(self.fig,self.anim_update, frames=sim_frames, blit=True, interval = interval)
        if show:
            plt.show()
        # self.fig.show()

    def save(self, filename):
        # gifWriter = PillowWriter(fps=30)
        # self.ani.save(filename, writer = gifWriter)
        writervideo = FFMpegWriter(fps=60)
        self.ani.save(filename, writer=writervideo, dpi=150)

    def set_title(self, title):
        self.ax.set_title(title)

    def _r_path_to_xy(self, r_path):
        coords = [self._r2c(r) for r in r_path]
        return coords

def package_path_gen(path, p_loc, d_loc):
    # make a path for the package picture
    pack_path = [p_loc]
    pack_state = 0
    for p1,p2 in zip(path[:-1], path[1:]):
        if pack_state == 0:
            pack_path.append(p_loc)
            if p1 == p2 == p_loc:
                pack_state = 1
        elif pack_state == 1:
            pack_path.append(p2)
            if p1 == p2 == d_loc:
                pack_state = 2
        elif pack_state == 2:
            pack_path.append(d_loc)
    return pack_path


def path_str_to_int(path_str):
    path_str_ls = path_str.split()
    path = [int(r) for r in path_str_ls]
    return path

def ansi_str_to_path_str(ansi_str):
    route = re.sub(r'\033\[(\d|;)+?m', '', ansi_str)
    foo = route.split('|')
    route = foo[0]
    route = route.replace('r', '')
    return route

def ansi_str_to_path_int(ansi_str):
    path_str = ansi_str_to_path_str(ansi_str)
    path_int = path_str_to_int(path_str)
    return path_int


def main():
    with open(FILE, 'r') as f:
        lines = f.readlines()
    route_color = lines[LINE]
    route = ansi_str_to_path_int(route_color)
    print(route_color)
    print(route)

    break_at = [0,6,9,10]

    ap = ArrowPlot(DIMS, ENV_INFO, route, break_idx=break_at)
    # ap.ax.annotate('', xy = (0.5,0.5), xytext=(1.5,1.5), arrowprops=dict(arrowstyle='simple', mutation_scale=30, color='black'))

    # arrow_seq = [
    #     'after',
    #     'after',
    #     'none',
    #     'before',
    #     'before',
    #     'none',
    #     'before',
    #     'after',
    #     'loop'
    # ]

    arr = ['after']*len(route)
    arr[-1] = 'none'
    arr[1] = 'loop'
    arr[3] = 'loop'
    arr[7] = 'loop'
    arr[12] = 'loop'
    # arr[20] = 'loop'

    ap.draw_path(arr)

    # ap.draw_path([0, 1, 2, 3, 7, 6, 5, 4, 9], arrow_seq=arrow_seq)
    # ap.draw_env(ENV_INFO)

    # ap.draw_arrow(9,10)
    # ap.draw_arrow(3,11)
    # ap.draw_arrow(4,13)
    ap.show()

def do_animation():
    with open(FILE, 'r') as f:
        lines = f.readlines()
    route_color = lines[LINE]
    route = ansi_str_to_path_int(route_color)
    print(route_color)
    print(route)

    anim = AnimPlot(DIMS, ENV_INFO, route, episode=3)
    anim.start()
    # anim.draw_env()
    # anim.start()


def save_learning_animation(file, vid_name, offset=0):
    with open(file, 'r') as f:
        lines = f.readlines()
    
    num = len(lines)
    mid = num // 2
    mid += offset
    # eps = [0,1,2,mid-1,mid,mid+1,num-3,num-2,num-1]
    eps = [50005, 50006, 50007]
    lns = [lines[i] for i in eps]
    for l in lns:
        print(l)
    routes = [ansi_str_to_path_int(l) for l in lns]

    for i,r in enumerate(routes):
        anim = AnimPlot(DIMS, ENV_INFO, r, episode=eps[i])
        anim.start(show=False)
        anim.save('{}_{}.mp4'.format(vid_name,eps[i]))

if __name__ == '__main__':
    # main()
    do_animation()
    # save_learning_animation('../output/mdp_trajectory_log.txt', '../vids/updated_new_ep')
