import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class TSviz:
    
    def __init__(self,columns,rows,num):
        self.xNum = columns
        self.yNum = rows
        self.num = num
        
        scale = 1.5
        
        self.fig, self.ax = plt.subplots(figsize = [columns*scale,rows*scale])
        
        # self.fig = plt.figure(num,figsize = [columns*scale,rows*scale])
        # self.ax = plt.axes()
        
        # shift labels to middle
        # xoffset = matplotlib.transforms.ScaledTranslation(0.4*scale, 0, self.fig.dpi_scale_trans)
        # yoffset = matplotlib.transforms.ScaledTranslation(0, 0.4*scale, self.fig.dpi_scale_trans)
        # for label in self.ax.xaxis.get_majorticklabels():
        #     label.set_transform(label.get_transform() + xoffset)
        # for label in self.ax.yaxis.get_majorticklabels():
        #     label.set_transform(label.get_transform() + yoffset)

        self.ax.axis([0,self.xNum,0,self.yNum])
        self.ax.grid(True)
        self.ax.set_xticks(list(range(0,self.xNum)))
        self.ax.set_yticks(list(range(0,self.yNum)))
        self.ax.invert_yaxis()

        # print(type(self.fig))
        # self.fig.savefig('ts' + str(num) +'.png')
            
    def drawRect(self,x,y,color, label = None):
        self.ax.fill([y+1, y+1, y, y],[x, x+1, x+1, x], color, label = label)
        
    def drawMarker(self, x,y,color):
        circle = plt.Circle((x+0.5,y+0.5), 0.1, color = color)
        self.ax.add_artist(circle)
        
    # draws arrow from coord 1 to coord 2
    def drawArrow(self, c1, c2, color):
        x = c2[0]
        y = c2[1]
        
        
        if (c2[0] - c1[0]) == 1:
            arrow = mpatches.Arrow(x-0.25,y+0.65,0.5,0, color = color, width = 0.4)
        elif (c2[0] - c1[0]) == -1:
            arrow = mpatches.Arrow(x+1.25,y+0.35,-0.5,0, color = color, width = 0.4)
        elif (c2[1] - c1[1]) == 1:
            arrow = mpatches.Arrow(x+0.35,y-0.25,0,0.5, color = color, width = 0.4)
        elif (c2[1] - c1[1]) == -1:
            arrow = mpatches.Arrow(x+0.65,y+1.25,0,-0.5, color = color, width = 0.4)
        elif (c1 == c2):
            arrow = mpatches.FancyArrowPatch((x+0.5,y+0.3),(x+0.5,y+0.7),color = color, 
                                             arrowstyle= "Simple, tail_width=0.7, head_width=4, head_length=8",
                                             connectionstyle="arc3,rad=1")
        else:
            raise Exception("bad arrow edge")
        self.ax.add_patch(arrow)
        
    def save(self):
        self.fig.savefig('ts' + str(self.num) +'.png')

    def show(self):
        self.ax.legend(loc='upper left')
        plt.show()

            

if __name__ == '__main__':

    viz = TSviz(4,4,22)
    viz.drawRect(3,0,'tab:blue', 'Initial')
    viz.drawRect(1,1,'tab:orange', 'Drop-off')
    viz.drawRect(3,3, 'yellow', 'Pick-up')
    viz.drawRect(1,3,'tab:cyan', 'Region 1')
    # viz.drawRect(2,3, 'tab:cyan')
    viz.drawRect(2,3, 'tab:purple', 'Region 2')
    # viz.drawRect(3,1, 'tab:purple')
    # viz.drawRect(1,2,'blue')
    viz.show()