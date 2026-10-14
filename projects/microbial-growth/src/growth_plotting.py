"""Pillow plotting primitives for retained workflow figures, without old runners."""
from PIL import Image, ImageDraw, ImageFont


def font_for_plot(size, bold=False):
    candidates = [
        '/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf',
        'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf',
        'arialbd.ttf' if bold else 'arial.ttf',
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path,size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def panel(im,d,txt,box,title,xlim,ylim,xticks,yticks,xlabel):
    left,top,right,bottom=box
    def xy(x,y):
        return left+(x-xlim[0])/(xlim[1]-xlim[0])*(right-left), bottom-(y-ylim[0])/(ylim[1]-ylim[0])*(bottom-top)
    txt(left,top-47,title,27,True)
    for y in yticks:
        yy=xy(xlim[0],y)[1];d.line((left,yy,right,yy),fill='#DFE5EB',width=1);txt(left-71,yy-12,f'{y:.2f}',20)
    for x in xticks:
        txt(xy(x,ylim[0])[0]-10,bottom+13,f'{x:g}',21)
    d.line((left,top,left,bottom,right,bottom),fill='#243249',width=2);txt(left+150,bottom+49,xlabel,22)
    def curve(x,y,color,lo=None,hi=None,dots=True):
        if lo is not None:
            overlay=Image.new('RGBA',im.size,(0,0,0,0));od=ImageDraw.Draw(overlay)
            rgb=tuple(int(color[j:j+2],16) for j in [1,3,5])
            od.polygon([xy(a,b) for a,b in zip(x,lo)]+[xy(a,b) for a,b in zip(x[::-1],hi[::-1])],fill=rgb+(28,))
            im.paste(overlay,(0,0),overlay)
        pts=[xy(a,b) for a,b in zip(x,y)];d.line(pts,fill=color,width=4)
        if dots:
            for a,b in pts:
                d.ellipse((a-4,b-4,a+4,b+4),fill=color)
    return curve
