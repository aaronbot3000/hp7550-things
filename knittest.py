import math

startx = 100
starty = 100
endx = -210
endy = 110

def dosomething():
  dx = endx - startx
  dy = endy - starty
  dd = max(abs(dx), abs(dy))
  
  for p in range(dd):
    t = p / float(dd)
    px = (endx - startx) * t + startx
    py = (endy - starty) * t + starty
    print(dy * px - dx * py + endx * starty - endy * startx)
    percent = 1 - abs((dy * px - dx * py + endx * starty - endy * startx) / math.sqrt(math.pow(dy, 2) + math.pow(dx, 2)))
    print(percent)

dosomething()
