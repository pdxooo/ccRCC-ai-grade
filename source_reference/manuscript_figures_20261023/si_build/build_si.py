"""Supplementary Information PDF from SI_final.md and the supplementary figure PNGs (reportlab). The markdown holds the
final text of the Supplementary Information; the figures are read from ../figures. Usage: python build_si.py OUT.pdf"""
import re,sys
from pathlib import Path
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether,PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.pdfgen import canvas

HERE=Path(__file__).resolve().parent
FIG=HERE.parent/'figures'
if not FIG.is_dir():
    FIG=next((p/'02_图件/补充图' for p in HERE.parents if (p/'02_图件/补充图').is_dir()),FIG)
import os
fd=Path(os.environ.get('SI_FONT_DIR','/usr/share/fonts/truetype/liberation2'))
font_files=(('Sans','LiberationSans-Regular.ttf'),('Sans-B','LiberationSans-Bold.ttf'),('Sans-I','LiberationSans-Italic.ttf'),('Sans-BI','LiberationSans-BoldItalic.ttf'),('Serif','LiberationSerif-Regular.ttf'),('Serif-B','LiberationSerif-Bold.ttf'),('Mono','LiberationMono-Regular.ttf'))
if not (fd/font_files[0][1]).exists():
    fd=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
    font_files=(('Sans','arial.ttf'),('Sans-B','arialbd.ttf'),('Sans-I','ariali.ttf'),('Sans-BI','arialbi.ttf'),('Serif','times.ttf'),('Serif-B','timesbd.ttf'),('Mono','cour.ttf'))
for nm,f in font_files:pdfmetrics.registerFont(TTFont(nm,str(fd/f)))
addMapping('Sans',0,0,'Sans');addMapping('Sans',1,0,'Sans-B');addMapping('Sans',0,1,'Sans-I');addMapping('Sans',1,1,'Sans-BI')
INK=colors.HexColor('#222222');MUTED=colors.HexColor('#6B6B6B');RULE=colors.HexColor('#D9D9D9');HEAD=colors.HexColor('#F2F2F0')
PW,PH=A4;M=20*mm;TW=PW-2*M
S=dict(title=ParagraphStyle('t',fontName='Serif',fontSize=20,leading=24,textColor=INK,spaceAfter=8),
       sub=ParagraphStyle('s',fontName='Sans-B',fontSize=10.5,leading=14,textColor=INK,spaceAfter=6),
       h2=ParagraphStyle('h',fontName='Serif',fontSize=13.5,leading=17,textColor=INK,spaceBefore=12,spaceAfter=5,keepWithNext=1),
       body=ParagraphStyle('b',fontName='Sans',fontSize=9.2,leading=12.6,textColor=INK,spaceAfter=5),
       bullet=ParagraphStyle('u',fontName='Sans',fontSize=9.2,leading=12.6,textColor=INK,spaceAfter=3,leftIndent=11,bulletIndent=2,bulletFontName='Sans'),
       cell=ParagraphStyle('c',fontName='Sans',fontSize=7.4,leading=9.2,textColor=INK),
       hcell=ParagraphStyle('hc',fontName='Sans',fontSize=7.4,leading=9.2,textColor=MUTED),
       legend=ParagraphStyle('l',fontName='Sans',fontSize=8.6,leading=11.6,textColor=INK,spaceBefore=4,spaceAfter=10))
SUP=dict(zip('⁰¹²³⁴⁵⁶⁷⁸⁹⁻','0123456789−'));SUB=dict(zip('₀₁₂₃₄₅₆₇₈₉','0123456789'))

def inline(t):
    t=t.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    t=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',t);t=re.sub(r'`([^`]+)`',r'<font name="Mono" size="8.4">\1</font>',t)
    t=re.sub('[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+',lambda m:'<super>'+''.join(SUP[c] for c in m.group(0))+'</super>',t)
    return re.sub('[₀₁₂₃₄₅₆₇₈₉]+',lambda m:'<sub>'+''.join(SUB[c] for c in m.group(0))+'</sub>',t)

def table(lines):
    rows=[[c.strip() for c in l.strip().strip('|').split('|')] for l in lines if not re.match(r'^\|[\s\-|]+\|$',l.strip())]
    n=len(rows[0]);assert all(len(r)==n for r in rows),lines[0]
    # column widths: every column is at least as wide as its longest unbreakable word, and the remaining width goes to the
    # columns whose text would otherwise wrap, in proportion to how much more they need
    sw=lambda t:pdfmetrics.stringWidth(t,'Sans',7.4)+6.5
    minw=[max(sw(t) for x in (r[j] for r in rows) for t in (x.split() or [''])) for j in range(n)]
    want=[min(max(sw(x) for x in (r[j] for r in rows)),0.45*TW) for j in range(n)]
    if sum(minw)>=TW:w=[TW*x/sum(minw) for x in minw]
    else:
        need=[max(a-b,0) for a,b in zip(want,minw)];extra=TW-sum(minw);tot=sum(need)
        w=[b+(extra*nd/tot if tot else extra/n) for b,nd in zip(minw,need)]
        if tot and extra>tot:w=[a+(extra-tot)*a/sum(want) for a in want]
    data=[[Paragraph(inline(c),S['hcell' if i==0 else 'cell']) for c in r] for i,r in enumerate(rows)]
    tb=Table(data,colWidths=w,repeatRows=1,hAlign='LEFT')
    tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HEAD),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),0.6,MUTED),
                            ('LINEBELOW',(0,1),(-1,-1),0.25,RULE),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),
                            ('TOPPADDING',(0,0),(-1,-1),2.2),('BOTTOMPADDING',(0,0),(-1,-1),2.2)]))
    return [tb,Spacer(1,7)]

def figure(path,legend):
    im=PILImage.open(path);w,h=im.size;wd=TW;ht=wd*h/w;room=PH-2*M-14*mm-58*mm  # leave room for the legend
    if ht>room:wd*=room/ht;ht=room
    return KeepTogether([Image(str(path),width=wd,height=ht),Paragraph(inline(legend),S['legend'])])

def story(md):
    blocks=[b.strip('\n') for b in re.split(r'\n\s*\n',md) if b.strip()];out=[];i=0
    while i<len(blocks):
        b=blocks[i];ls=b.split('\n')
        if b.startswith('# '):out.append(Paragraph(inline(b[2:]),S['title']))
        elif b.startswith('## '):
            if b[3:].startswith('Supplementary Figures'):out.append(PageBreak())
            out.append(Paragraph(inline(b[3:]),S['h2']))
        elif b.startswith('!['):
            p=re.match(r'!\[[^\]]*\]\(([^)]+)\)',b).group(1);out.append(figure(FIG/p,blocks[i+1].replace('\n',' ')));i+=1
        elif b.startswith('|'):out+=table(ls)
        elif all(l.startswith('- ') for l in ls):out+=[Paragraph(inline(l[2:]),S['bullet'],bulletText='•') for l in ls]
        elif i==1 and b.startswith('**'):out.append(Paragraph(inline(b.replace('**','')),S['sub']))
        else:out.append(Paragraph(inline(' '.join(ls)),S['body']))
        i+=1
    return out

class Numbered(canvas.Canvas):
    """Header and 'Page n of N' footer, written once the page count is known."""
    def __init__(self,*a,**k):super().__init__(*a,**k);self._pages=[]
    def showPage(self):self._pages.append(dict(self.__dict__));self._startPage()
    def save(self):
        n=len(self._pages)
        for st in self._pages:
            self.__dict__.update(st);self.setFont('Sans',7);self.setFillColor(MUTED)
            self.drawString(M,PH-12*mm,'Supplementary Information');self.drawRightString(PW-M,10*mm,f'Page {self._pageNumber} of {n}')
            super().showPage()
        super().save()

if __name__=='__main__':
    out=sys.argv[1];md=(HERE/'SI_final.md').read_text(encoding='utf-8')
    doc=BaseDocTemplate(out,pagesize=A4,leftMargin=M,rightMargin=M,topMargin=M,bottomMargin=M,title='Supplementary Information',
                        author='',subject=re.search(r'^\*\*(.+)\*\*$',md,re.M).group(1))
    doc.addPageTemplates([PageTemplate('p',[Frame(M,M,TW,PH-2*M,id='f',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)])])
    doc.build(story(md),canvasmaker=Numbered);print('WROTE',out,flush=True)
