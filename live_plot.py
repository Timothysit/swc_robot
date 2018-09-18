import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from matplotlib import style
# import seaborn as sns
# style.use('fivethirtyeight')
# style.use('seaborn-white')
style.use('bmh')

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
def animate(i):
	graph_data = open('blue_score.txt', 'r').read()
	lines = graph_data.split('\n')
	xs = [] 
	ys = [] 
	for line in lines:
		if len(line) > 1:
			x, y = line.split(',')
			xs.append(float(x))
			ys.append(float(y))
	# ax1.set_xlim([min(xs), max(xs)])
	ax1.clear()
	ax1.plot(xs, ys)
	# ax1.set_ylim([0,100])
	ax1.set_xlabel('Frames')
	ax1.set_ylabel('Blue score')
	# ax1.plot(ys, xs)

## Usage example: 
ani = animation.FuncAnimation(fig, animate, interval = 1000)
plt.show()
