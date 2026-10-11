#!/usr/bin/env python3
"""
Voxel Engine Testing Application v2.0
PySide6 + OpenGL voxel algorithm testing & visualization tool.

Controls:
    LMB Drag  = Orbit camera
    RMB Drag  = Pan camera
    Wheel     = Zoom in/out
    MMB Drag  = Dolly
    Space     = Toggle auto-rotate
    Ctrl+Z    = Undo
    Ctrl+Y    = Redo
    Ctrl+S    = Screenshot
    G         = Generate
    F         = Toggle panel visibility

Install:  pip install PySide6 PyOpenGL numpy
"""

from __future__ import annotations

import sys, math, time, random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Any

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QLabel, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QScrollArea, QMessageBox, QSplitter,
    QToolBar, QFileDialog, QStatusBar,
)
from PySide6.QtCore import Qt, QTimer, Signal, QSize
from PySide6.QtGui import QKeySequence, QShortcut, QAction
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *


# ================================================================
#  Block Types
# ================================================================
class Block(IntEnum):
    AIR=0; STONE=1; DIRT=2; GRASS=3; SAND=4
    WATER=5; WOOD=6; LEAVES=7; SNOW=8; CUSTOM=255

BLOCK_NAMES = {
    Block.STONE:"Stone", Block.DIRT:"Dirt", Block.GRASS:"Grass",
    Block.SAND:"Sand", Block.WATER:"Water", Block.WOOD:"Wood",
    Block.LEAVES:"Leaves", Block.SNOW:"Snow", Block.CUSTOM:"Custom",
}

BLOCK_COLORS = {
    Block.AIR:(0,0,0,0), Block.STONE:(.58,.58,.58,1), Block.DIRT:(.62,.42,.22,1),
    Block.GRASS:(.28,.78,.24,1), Block.SAND:(.92,.89,.58,1), Block.WATER:(.22,.45,.92,.7),
    Block.WOOD:(.52,.32,.12,1), Block.LEAVES:(.18,.60,.14,1), Block.SNOW:(.94,.94,1.,1),
    Block.CUSTOM:(1.,0.,1.,1),
}


# ================================================================
#  Perlin Noise
# ================================================================
class PerlinNoise:
    def __init__(self, seed=0):
        rng = random.Random(seed)
        p = list(range(256)); rng.shuffle(p)
        self.perm = p + p

    @staticmethod
    def _fade(t): return t*t*t*(t*(t*6-15)+10)
    @staticmethod
    def _lerp(t, a, b): return a + t*(b-a)
    @staticmethod
    def _grad(h, x, y, z):
        h &= 15
        u = x if h<8 else y
        v = y if h<4 else (x if h in(12,14) else z)
        return (-u if h&1 else u)+(-v if h&2 else v)

    def noise3d(self, x, y, z):
        p = self.perm
        X=int(math.floor(x))&255; Y=int(math.floor(y))&255; Z=int(math.floor(z))&255
        x-=math.floor(x); y-=math.floor(y); z-=math.floor(z)
        u,v,w = self._fade(x),self._fade(y),self._fade(z)
        A=p[X]+Y; AA=p[A]+Z; AB=p[A+1]+Z
        B=p[X+1]+Y; BA=p[B]+Z; BB=p[B+1]+Z
        return self._lerp(w,
            self._lerp(v,
                self._lerp(u,self._grad(p[AA],x,y,z),self._grad(p[BA],x-1,y,z)),
                self._lerp(u,self._grad(p[AB],x,y-1,z),self._grad(p[BB],x-1,y-1,z))),
            self._lerp(v,
                self._lerp(u,self._grad(p[AA+1],x,y,z-1),self._grad(p[BA+1],x-1,y,z-1)),
                self._lerp(u,self._grad(p[AB+1],x,y-1,z-1),self._grad(p[BB+1],x-1,y-1,z-1))))

    def octave(self, x, y, z, octaves=4, persistence=0.5, lacunarity=2.0):
        total=0.; max_a=0.; freq=1.; amp=1.
        for _ in range(octaves):
            total+=self.noise3d(x*freq,y*freq,z*freq)*amp; max_a+=amp
            amp*=persistence; freq*=lacunarity
        return total/max_a if max_a>0 else 0.


# ================================================================
#  Voxel Chunk
# ================================================================
class VoxelChunk:
    FACES = [
        ((1,0,0),[(1,0,0),(1,1,0),(1,1,1),(1,0,1)]),
        ((-1,0,0),[(0,0,1),(0,1,1),(0,1,0),(0,0,0)]),
        ((0,1,0),[(0,1,1),(1,1,1),(1,1,0),(0,1,0)]),
        ((0,-1,0),[(0,0,0),(1,0,0),(1,0,1),(0,0,1)]),
        ((0,0,1),[(1,0,1),(0,0,1),(0,1,1),(1,1,1)]),
        ((0,0,-1),[(0,0,0),(1,0,0),(1,1,0),(0,1,0)]),
    ]
    NBR = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    TRI_IDX = ((0,1,2),(0,2,3))
    DIR_SHADE = [0.55+0.45*max(0.,n[0]*.3+n[1]*.6+n[2]*.1+.25) for n,_ in FACES]

    def __init__(self, size=32):
        self.size = size
        self.data = np.zeros((size,size,size), dtype=np.uint8)
        self.colors: dict[tuple,int] = {}
        self.mesh_dirty = True
        self.vertices = np.zeros((0,3),dtype=np.float32)
        self.normals = np.zeros((0,3),dtype=np.float32)
        self.colors_arr = np.zeros((0,4),dtype=np.float32)
        self.vertex_count = 0; self.face_count = 0

    def get(self,x,y,z):
        if 0<=x<self.size and 0<=y<self.size and 0<=z<self.size: return int(self.data[x,y,z])
        return 0

    def set(self,x,y,z,v):
        if 0<=x<self.size and 0<=y<self.size and 0<=z<self.size:
            self.data[x,y,z]=v; self.mesh_dirty=True

    def get_color(self,x,y,z):
        if (x,y,z) in self.colors: return self.colors[(x,y,z)]
        return BLOCK_COLORS.get(int(self.data[x,y,z]),(1,0,1,1))

    def set_color(self,x,y,z,c):
        if self.data[x,y,z]: self.colors[(x,y,z)]=c; self.mesh_dirty=True

    def clear(self):
        self.data.fill(0); self.colors.clear(); self.mesh_dirty=True

    def count_voxels(self): return int(np.count_nonzero(self.data))

    def count_by_type(self):
        r = {}
        for bt in np.unique(self.data):
            if bt > 0: r[int(bt)] = int(np.count_nonzero(self.data==bt))
        return r

    def clone(self):
        c = VoxelChunk(self.size)
        c.data = self.data.copy(); c.colors = dict(self.colors)
        c.mesh_dirty = True; return c

    def build_mesh(self):
        s = self.size
        occupied = np.argwhere(self.data != 0)
        if len(occupied) == 0:
            self.vertices=np.zeros((0,3),dtype=np.float32)
            self.normals=np.zeros((0,3),dtype=np.float32)
            self.colors_arr=np.zeros((0,4),dtype=np.float32)
            self.vertex_count=0; self.face_count=0; self.mesh_dirty=False; return

        VL=[]; NL=[]; CL=[]
        faces=self.FACES; nbrs=self.NBR; tri=self.TRI_IDX; shade=self.DIR_SHADE
        data=self.data; cmap=self.colors

        for idx in range(len(occupied)):
            x,y,z = int(occupied[idx,0]),int(occupied[idx,1]),int(occupied[idx,2])
            bc = cmap.get((x,y,z)) or BLOCK_COLORS.get(int(data[x,y,z]),(1,0,1,1))
            for fi in range(6):
                nx2,ny2,nz2 = x+nbrs[fi][0], y+nbrs[fi][1], z+nbrs[fi][2]
                if 0<=nx2<s and 0<=ny2<s and 0<=nz2<s and data[nx2,ny2,nz2]: continue
                nrm=faces[fi][0]; fv=faces[fi][1]; sh=shade[fi]
                sc=(bc[0]*sh,bc[1]*sh,bc[2]*sh,bc[3])
                for i0,i1,i2 in tri:
                    for vi in (i0,i1,i2):
                        p=fv[vi]
                        VL.append((x+p[0],y+p[1],z+p[2]))
                        NL.append(nrm); CL.append(sc)

        self.vertex_count=len(VL); self.face_count=self.vertex_count//3
        if VL:
            self.vertices=np.array(VL,dtype=np.float32)
            self.normals=np.array(NL,dtype=np.float32)
            self.colors_arr=np.array(CL,dtype=np.float32)
        else:
            self.vertices=np.zeros((0,3),dtype=np.float32)
            self.normals=np.zeros((0,3),dtype=np.float32)
            self.colors_arr=np.zeros((0,4),dtype=np.float32)
        self.mesh_dirty=False


# ================================================================
#  Undo Stack
# ================================================================
class UndoStack:
    MAX=40
    def __init__(self): self._u=[]; self._r=[]
    def push(self,c):
        self._u.append(c.clone()); self._r.clear()
        if len(self._u)>self.MAX: self._u.pop(0)
    def undo(self,c):
        if not self._u: return None
        self._r.append(c.clone()); return self._u.pop()
    def redo(self,c):
        if not self._r: return None
        self._u.append(c.clone()); return self._r.pop()
    def can_undo(self): return len(self._u)>0
    def can_redo(self): return len(self._r)>0
    def clear(self): self._u.clear(); self._r.clear()


# ================================================================
#  Algorithm Registry
# ================================================================
@dataclass
class AlgoInfo:
    name: str; fn: Callable; params: dict; tooltip: str=""; category: str="Generator"

def _reg_algos() -> dict[str, AlgoInfo]:
    R = {}

    def _flat(c, seed, p):
        c.clear(); h=min(int(p.get("height",4)),c.size); b=p.get("block_type",Block.DIRT)
        for x in range(c.size):
            for z in range(c.size):
                for y in range(h): c.set(x,y,z,b)
    R["Flat Terrain"]=AlgoInfo("Flat Terrain",_flat,{
        "height":{"label":"Height","type":int,"default":4,"min":1,"max":64,"step":1,"tooltip":"Solid layers from bottom"},
        "block_type":{"label":"Block Type","type":"combo","default":Block.DIRT,
            "options":{k:v for k,v in BLOCK_NAMES.items() if k!=Block.AIR},"tooltip":"Block material"},
    },"Fill bottom layers with solid block","Terrain")

    def _heightmap(c, seed, p):
        c.clear(); hs=p.get("height_scale",10); ns=p.get("noise_scale",.05)
        pn=PerlinNoise(seed)
        for x in range(c.size):
            for z in range(c.size):
                h=max(1,min(int((pn.octave(x*ns,0,z*ns,4)+1)*.5*hs)+1,c.size-1))
                for y in range(h):
                    c.set(x,y,z, Block.GRASS if y==h-1 else (Block.DIRT if y>h-4 else Block.STONE))
    R["Perlin Heightmap"]=AlgoInfo("Perlin Heightmap",_heightmap,{
        "height_scale":{"label":"Height Scale","type":float,"default":10,"min":1,"max":50,"step":1,"tooltip":"Max terrain height"},
        "noise_scale":{"label":"Noise Scale","type":float,"default":.05,"min":.01,"max":.3,"step":.005,"tooltip":"Noise frequency (lower=larger features)"},
    },"Octave Perlin noise terrain","Terrain")

    def _ds(c, seed, p):
        c.clear(); rng=random.Random(seed); s=c.size
        # Diamond-square needs 2^n+1 grid; find largest valid size <= s
        import math as _m
        gs = 2**int(_m.log2(max(s-1, 2))) + 1
        if gs > s: gs = (gs - 1) // 2 + 1
        if gs < 3: gs = 3
        hm=np.zeros((gs,gs)); step=gs-1; sc=.6
        hm[0,0]=rng.random(); hm[0,step]=rng.random()
        hm[step,0]=rng.random(); hm[step,step]=rng.random()
        while step>1:
            h=step//2
            # Diamond step: fill centers of squares
            for y in range(0, gs-1, step):
                for x in range(0, gs-1, step):
                    hm[y+h,x+h]=(hm[y,x]+hm[y,x+step]+hm[y+step,x]+hm[y+step,x+step])/4+(rng.random()-.5)*sc
            # Square step: fill centers of diamonds
            for y in range(0, gs, h):
                for x in range(((y//h)%2)*h, gs, step):
                    if hm[y,x]!=0: continue
                    t=n=0
                    if y>=h: t+=hm[y-h,x]; n+=1
                    if y+h<gs: t+=hm[y+h,x]; n+=1
                    if x>=h: t+=hm[y,x-h]; n+=1
                    if x+h<gs: t+=hm[y,x+h]; n+=1
                    if n>0: hm[y,x]=t/n+(rng.random()-.5)*sc
            step=h; sc*=.5
        hm=hm[:s,:s]
        gs2=hm.shape[0]  # actual size after crop
        hm=(hm-hm.min())/(hm.max()-hm.min()+1e-9); hs=gs2//3
        for x in range(gs2):
            for z in range(gs2):
                h2=max(1,min(int(hm[z,x]*hs)+1,c.size-1))
                for y in range(h2):
                    c.set(x,y,z, Block.GRASS if y==h2-1 else (Block.DIRT if y>h2-3 else Block.STONE))
    R["Diamond-Square"]=AlgoInfo("Diamond-Square",_ds,{},"Diamond-square terrain","Terrain")

    def _sphere(c, seed, p):
        c.clear(); r=p.get("radius",c.size//2-1); cx=cy=cz=c.size//2
        for x in range(c.size):
            for y in range(c.size):
                for z in range(c.size):
                    if (x-cx)**2+(y-cy)**2+(z-cz)**2<=r*r: c.set(x,y,z,Block.GRASS)
    R["Sphere"]=AlgoInfo("Sphere",_sphere,{
        "radius":{"label":"Radius","type":int,"default":15,"min":2,"max":32,"step":1,"tooltip":"Sphere radius in voxels"},
    },"Filled sphere","Shapes")

    def _torus(c, seed, p):
        c.clear(); R=p.get("major_radius",c.size//4); r=p.get("minor_radius",c.size//8)
        cx=cy=cz=c.size//2
        for x in range(c.size):
            for y in range(c.size):
                for z in range(c.size):
                    dx,dy,dz=x-cx,y-cy,z-cz
                    dxz=math.sqrt(dx*dx+dz*dz); d=math.sqrt((dxz-R)**2+dy*dy)
                    if d<=r:
                        t=d/max(r,1); c.set(x,y,z,Block.CUSTOM)
                        c.set_color(x,y,z,(.3+.5*(1-t),.5*t,.8*(1-t)+.2,1.))
    R["Torus"]=AlgoInfo("Torus",_torus,{
        "major_radius":{"label":"Major Radius","type":int,"default":8,"min":3,"max":20,"step":1,"tooltip":"Center-to-tube distance"},
        "minor_radius":{"label":"Minor Radius","type":int,"default":4,"min":1,"max":12,"step":1,"tooltip":"Tube radius"},
    },"Doughnut-shaped torus","Shapes")

    def _checker(c, seed, p):
        c.clear()
        for x in range(c.size):
            for y in range(c.size):
                for z in range(c.size):
                    if (x+y+z)%2==0: c.set(x,y,z,Block.STONE)
    R["Checkerboard"]=AlgoInfo("Checkerboard",_checker,{},"3D alternating pattern","Shapes")

    def _menger(c, seed, p):
        c.clear(); lv=2 if c.size>=27 else 1; ss=3**lv; sc=c.size/ss
        def fill(x,y,z,l):
            if l==0: return True
            s2=3**(l-1); mx=(x//s2)%3==1; my=(y//s2)%3==1; mz=(z//s2)%3==1
            if (mx and my)or(mx and mz)or(my and mz): return False
            return fill(x%s2,y%s2,z%s2,l-1)
        for x in range(c.size):
            for y in range(c.size):
                for z in range(c.size):
                    sx,sy,sz=int(x/sc),int(y/sc),int(z/sc)
                    if sx<ss and sy<ss and sz<ss and fill(sx,sy,sz,lv):
                        d=sum(1 for l in range(lv) for a in [(sx//(3**l))%3==1,(sy//(3**l))%3==1,(sz//(3**l))%3==1] if a)
                        t=d/max(lv*3,1); c.set(x,y,z,Block.CUSTOM)
                        c.set_color(x,y,z,(.85-.5*t,.25+.45*t,.82-.3*t,1.))
    R["Menger Sponge"]=AlgoInfo("Menger Sponge",_menger,{},"Menger sponge fractal","Shapes")

    def _n3d(c, seed, p):
        c.clear(); thr=p.get("threshold",0.); ns=p.get("noise_scale",.08)
        pn=PerlinNoise(seed)
        for x in range(c.size):
            for y in range(c.size):
                for z in range(c.size):
                    n=pn.octave(x*ns,y*ns,z*ns,3)
                    if n>thr:
                        t=(n-thr)/(1-thr+1e-9); c.set(x,y,z,Block.CUSTOM)
                        c.set_color(x,y,z,(.2+.7*t,.35+.45*(1-t),.85-.55*t,1.))
    R["3D Perlin Noise"]=AlgoInfo("3D Perlin Noise",_n3d,{
        "threshold":{"label":"Threshold","type":float,"default":0.,"min":-.8,"max":.8,"step":.05,"tooltip":"Noise cutoff"},
        "noise_scale":{"label":"Noise Scale","type":float,"default":.08,"min":.01,"max":.3,"step":.005,"tooltip":"Noise frequency"},
    },"3D noise iso-surface","Noise")

    def _scatter(c, seed, p):
        c.clear(); d=p.get("density",.1); rng=random.Random(seed)
        ts=[Block.STONE,Block.DIRT,Block.GRASS,Block.SAND,Block.WOOD,Block.LEAVES]
        for x in range(c.size):
            for y in range(c.size):
                for z in range(c.size):
                    if rng.random()<d: c.set(x,y,z,rng.choice(ts))
    R["Random Scatter"]=AlgoInfo("Random Scatter",_scatter,{
        "density":{"label":"Density","type":float,"default":.1,"min":.01,"max":.5,"step":.01,"tooltip":"Fill probability per voxel"},
    },"Random block distribution","Noise")

    def _wave(c, seed, p):
        c.clear(); amp=p.get("amplitude",4); freq=p.get("frequency",.3)
        cy=c.size//2
        for x in range(c.size):
            for z in range(c.size):
                h=max(0,min(int(cy+amp*math.sin(x*freq)*math.cos(z*freq)),c.size-1))
                for y in range(h+1): c.set(x,y,z,Block.SAND)
    R["Sine Wave"]=AlgoInfo("Sine Wave",_wave,{
        "amplitude":{"label":"Amplitude","type":float,"default":4,"min":1,"max":20,"step":1,"tooltip":"Wave height"},
        "frequency":{"label":"Frequency","type":float,"default":.3,"min":.05,"max":1.,"step":.05,"tooltip":"Spatial frequency"},
    },"Sine/cosine wave surface","Noise")

    def _worm(c, seed, p):
        c.clear(); c.data[:]=Block.STONE
        worms=p.get("worms",5); length=p.get("length",60); radius=p.get("radius",2)
        rng=random.Random(seed)
        for _ in range(worms):
            wx=rng.randint(radius,c.size-radius-1); wy=rng.randint(c.size//3,2*c.size//3)
            wz=rng.randint(radius,c.size-radius-1)
            dx,dy,dz=rng.choice([-1,0,1]),rng.choice([-1,0,1]),rng.choice([-1,0,1])
            for _ in range(length):
                for ox in range(-radius,radius+1):
                    for oy in range(-radius,radius+1):
                        for oz in range(-radius,radius+1):
                            if ox*ox+oy*oy+oz*oz<=radius*radius:
                                nx,ny,nz=wx+ox,wy+oy,wz+oz
                                if 0<=nx<c.size and 0<=ny<c.size and 0<=nz<c.size:
                                    c.data[nx,ny,nz]=0
                dx+=rng.choice([-1,0,0,0,1]);dy+=rng.choice([-1,0,0,0,1]);dz+=rng.choice([-1,0,0,0,1])
                dx=max(-1,min(1,dx));dy=max(-1,min(1,dy));dz=max(-1,min(1,dz))
                wx+=dx;wy+=dy;wz+=dz
                wx=max(radius,min(c.size-radius-1,wx))
                wy=max(radius,min(c.size-radius-1,wy))
                wz=max(radius,min(c.size-radius-1,wz))
        c.mesh_dirty=True
    R["Worm Cave"]=AlgoInfo("Worm Cave",_worm,{
        "worms":{"label":"Worms","type":int,"default":5,"min":1,"max":30,"step":1,"tooltip":"Number of tunnels"},
        "length":{"label":"Length","type":int,"default":60,"min":10,"max":200,"step":10,"tooltip":"Steps per worm"},
        "radius":{"label":"Radius","type":int,"default":2,"min":1,"max":5,"step":1,"tooltip":"Tunnel radius"},
    },"Worm-tunnel cave system","Noise")

    def _arches(c, seed, p):
        c.clear(); ac=p.get("arches",3); rng=random.Random(seed)
        for _ in range(ac):
            cx=rng.randint(c.size//4,3*c.size//4); cz=rng.randint(c.size//4,3*c.size//4)
            r=rng.randint(c.size//6,c.size//4); h=rng.randint(c.size//3,c.size//2)
            for x in range(c.size):
                for y in range(c.size):
                    for z in range(c.size):
                        dx,dz=x-cx,z-cz; dxz=math.sqrt(dx*dx+dz*dz)
                        if dxz<=1.5:
                            if 0<=y<h: c.set(x,y,z,Block.STONE)
                        dt=math.sqrt(dxz*dxz+(y-h)*(y-h))
                        if dxz>1.5 and dt<=r*.8: c.set(x,y,z,Block.SAND)
    R["Arches"]=AlgoInfo("Arches",_arches,{
        "arches":{"label":"Arches","type":int,"default":3,"min":1,"max":8,"step":1,"tooltip":"Number of arches"},
    },"Stone arch structures","Structures")

    def _spiral(c, seed, p):
        c.clear(); cx=cy=cz=c.size//2; mr=c.size//2-2
        turns=p.get("turns",3); th=p.get("thickness",2); steps=turns*100
        for i in range(steps):
            t=i/steps; a=t*turns*2*math.pi; r=mr*t
            x=int(cx+r*math.cos(a)); y=int(cy+t*(c.size-4)); z=int(cz+r*math.sin(a))
            for ox in range(-th,th+1):
                for oy in range(-th//2,th//2+1):
                    for oz in range(-th,th+1):
                        if ox*ox+oy*oy+oz*oz<=th*th:
                            nx,ny,nz=x+ox,y+oy,z+oz
                            if 0<=nx<c.size and 0<=ny<c.size and 0<=nz<c.size:
                                c.set(nx,ny,nz,Block.CUSTOM)
                                c.set_color(nx,ny,nz,(t,.5,1.-t,1.))
    R["Spiral Tower"]=AlgoInfo("Spiral Tower",_spiral,{
        "turns":{"label":"Turns","type":int,"default":3,"min":1,"max":10,"step":1,"tooltip":"Spiral rotations"},
        "thickness":{"label":"Thickness","type":int,"default":2,"min":1,"max":5,"step":1,"tooltip":"Strand thickness"},
    },"Spiral tower structure","Structures")

    def _pyramid(c, seed, p):
        c.clear(); cx=cz=c.size//2
        for y in range(c.size):
            half=max(0,(c.size-1-y)//2)
            for x in range(cx-half,cx+half+1):
                for z in range(cz-half,cz+half+1):
                    if 0<=x<c.size and 0<=y<c.size and 0<=z<c.size:
                        t=y/max(c.size-1,1); c.set(x,y,z,Block.SAND)
                        c.set_color(x,y,z,(.9-.4*t,.75-.3*t,.4+.2*t,1.))
    R["Pyramid"]=AlgoInfo("Pyramid",_pyramid,{},"Stepped pyramid","Structures")

    def _twt(c, seed, p):
        _heightmap(c, seed, p); tc=p.get("tree_count",15); rng=random.Random(seed+1)
        for _ in range(tc):
            tx=rng.randint(3,c.size-4);tz=rng.randint(3,c.size-4);ty=c.size-1
            while ty>0 and c.get(tx,ty,tz)==0: ty-=1
            if ty<=0: continue
            h=rng.randint(4,7)
            for y in range(h): c.set(tx,ty+1+y,tz,Block.WOOD)
            for dx in range(-2,3):
                for dy in range(-1,3):
                    for dz in range(-2,3):
                        if abs(dx)+abs(dy)+abs(dz)<4:
                            nx,ny,nz=tx+dx,ty+1+h+dy,tz+dz
                            if 0<=nx<c.size and 0<=ny<c.size and 0<=nz<c.size and c.get(nx,ny,nz)==0:
                                c.set(nx,ny,nz,Block.LEAVES)
    R["Terrain + Trees"]=AlgoInfo("Terrain + Trees",_twt,{
        "height_scale":{"label":"Height Scale","type":float,"default":10,"min":1,"max":50,"step":1,"tooltip":"Max terrain height"},
        "noise_scale":{"label":"Noise Scale","type":float,"default":.05,"min":.01,"max":.3,"step":.005,"tooltip":"Noise frequency"},
        "tree_count":{"label":"Tree Count","type":int,"default":15,"min":1,"max":80,"step":1,"tooltip":"Number of trees"},
    },"Heightmap with trees","Structures")

    return R


# ================================================================
#  Operation Registry
# ================================================================
@dataclass
class OpInfo:
    name: str; fn: Callable; params: dict = field(default_factory=dict); tooltip: str=""

def _reg_ops() -> dict[str, OpInfo]:
    R = {}

    def _erosion(c, seed, p):
        iters=p.get("iterations",5)
        for _ in range(iters):
            ch=[]
            for x in range(1,c.size-1):
                for z in range(1,c.size-1):
                    y=c.size-1
                    while y>0 and c.get(x,y,z)==0: y-=1
                    if y<=0: continue
                    for dx,dz in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny=c.size-1
                        while ny>0 and c.get(x+dx,ny,z+dz)==0: ny-=1
                        if y-ny>1: ch.append((x,y,z,x+dx,ny+1,z+dz))
            for sx,sy,sz,dx,dy,dz in ch:
                b=c.get(sx,sy,sz);c.set(sx,sy,sz,0);c.set(dx,dy,dz,b)
    R["Erosion"]=OpInfo("Erosion",_erosion,{
        "iterations":{"label":"Iterations","type":int,"default":5,"min":1,"max":50,"step":1,"tooltip":"Erosion passes"},
    },"Hydraulic erosion simulation")

    def _smooth(c, seed, p):
        o=c.data.copy()
        for x in range(1,c.size-1):
            for y in range(1,c.size-1):
                for z in range(1,c.size-1):
                    n=int(np.count_nonzero(o[x-1:x+2,y-1:y+2,z-1:z+2]))-int(o[x,y,z]>0)
                    if o[x,y,z]==0 and n>=20: c.set(x,y,z,Block.DIRT)
                    elif o[x,y,z]>0 and n<=3: c.set(x,y,z,0)
    R["Smooth"]=OpInfo("Smooth",_smooth,{},"Fill gaps and remove outliers")

    def _ff(c, seed, p):
        cx=cz=c.size//2;cy=0
        while cy<c.size and c.get(cx,cy,cz)==0: cy+=1
        if cy<c.size:
            old=c.get(cx,cy,cz);stk=[(cx,cy,cz)];vis=set()
            while stk:
                px,py,pz=stk.pop()
                if (px,py,pz) in vis: continue
                if not(0<=px<c.size and 0<=py<c.size and 0<=pz<c.size): continue
                if c.get(px,py,pz)!=old: continue
                vis.add((px,py,pz));c.set(px,py,pz,Block.GRASS)
                stk.extend([(px+1,py,pz),(px-1,py,pz),(px,py+1,pz),(px,py-1,pz),(px,py,pz+1),(px,py,pz-1)])
    R["Flood Fill (Center -> Grass)"]=OpInfo("Flood Fill (Center -> Grass)",_ff,{},"Flood-fill center with grass")

    def _ch(c, seed, p):
        for x in range(c.size):
            for y in range(c.size):
                for z in range(c.size):
                    if c.data[x,y,z]:
                        t=y/max(c.size-1,1); c.set_color(x,y,z,(.1+.9*t,.82-.6*t,.2+.3*(1-t),1.))
    R["Color by Height"]=OpInfo("Color by Height",_ch,{},"Green-to-red height gradient")

    def _cn(c, seed, p):
        pn=PerlinNoise(seed)
        for x in range(c.size):
            for y in range(c.size):
                for z in range(c.size):
                    if c.data[x,y,z]:
                        n=(pn.noise3d(x*.12,y*.12,z*.12)+1)*.5
                        c.set_color(x,y,z,(.3+.7*n,.2+.5*(1-n),.3+.4*abs(n-.5)*2,1.))
    R["Color by Noise"]=OpInfo("Color by Noise",_cn,{},"Perlin noise texture recolor")

    def _at(c, seed, p):
        tc=p.get("tree_count",10);rng=random.Random(seed)
        for _ in range(tc):
            tx=rng.randint(3,c.size-4);tz=rng.randint(3,c.size-4);ty=c.size-1
            while ty>0 and c.get(tx,ty,tz)==0: ty-=1
            if ty<=0: continue
            h=rng.randint(4,7)
            for y in range(h): c.set(tx,ty+1+y,tz,Block.WOOD)
            for dx in range(-2,3):
                for dy in range(-1,3):
                    for dz in range(-2,3):
                        if abs(dx)+abs(dy)+abs(dz)<4:
                            nx,ny,nz=tx+dx,ty+1+h+dy,tz+dz
                            if 0<=nx<c.size and 0<=ny<c.size and 0<=nz<c.size and c.get(nx,ny,nz)==0:
                                c.set(nx,ny,nz,Block.LEAVES)
    R["Add Trees"]=OpInfo("Add Trees",_at,{
        "tree_count":{"label":"Tree Count","type":int,"default":10,"min":1,"max":80,"step":1,"tooltip":"Trees to add"},
    },"Place random trees on terrain")

    def _inv(c, seed, p):
        m=c.data==0;c.data[m]=Block.STONE;c.data[~m]=0;c.mesh_dirty=True
    R["Invert"]=OpInfo("Invert",_inv,{},"Swap solid/empty")

    def _sub(c, seed, p):
        f=2;ns=max(4,c.size//f);nc=VoxelChunk(ns)
        for x in range(ns):
            for y in range(ns):
                for z in range(ns): nc.data[x,y,z]=c.data[x*f,y*f,z*f]
        c.data=nc.data;c.size=nc.size;c.colors.clear();c.mesh_dirty=True
    R["Subsample (2x)"]=OpInfo("Subsample (2x)",_sub,{},"Downsample by 2x")

    def _ctl(c, seed, p):
        for y in range(c.size-1,-1,-1):
            found=False
            for x in range(c.size):
                for z in range(c.size):
                    if c.data[x,y,z]>0: c.set(x,y,z,0);found=True
            if found: break
    R["Clear Top Layer"]=OpInfo("Clear Top Layer",_ctl,{},"Remove topmost voxel layer")

    return R


# ================================================================
#  CSG
# ================================================================
def _csg_union(a,b):
    r=VoxelChunk(a.size);r.data=np.maximum(a.data,b.data);r.colors={**b.colors,**a.colors};r.mesh_dirty=True;return r
def _csg_isect(a,b):
    r=VoxelChunk(a.size);m=(a.data>0)&(b.data>0)
    r.data=np.where(m,a.data,0).astype(np.uint8)
    ks=set(map(tuple,np.argwhere(m)));r.colors={k:v for k,v in a.colors.items() if k in ks};r.mesh_dirty=True;return r
def _csg_diff(a,b):
    r=VoxelChunk(a.size);m=b.data==0
    r.data=np.where(m,a.data,0).astype(np.uint8)
    ks=set(map(tuple,np.argwhere(m)));r.colors={k:v for k,v in a.colors.items() if k in ks};r.mesh_dirty=True;return r
def _make_shape(size,shape):
    c=VoxelChunk(size);cx=cy=cz=size//2;r=size//4
    for x in range(size):
        for y in range(size):
            for z in range(size):
                dx,dy,dz=x-cx,y-cy,z-cz;ok=False
                if shape=="Sphere": ok=dx*dx+dy*dy+dz*dz<=r*r
                elif shape=="Cube": ok=abs(dx)<=r and abs(dy)<=r and abs(dz)<=r
                elif shape=="Cylinder": ok=dx*dx+dz*dz<=r*r and abs(dy)<=r
                if ok: c.set(x,y,z,Block.SAND)
    return c


# ================================================================
#  OpenGL Viewport
# ================================================================
class VoxelViewport(QOpenGLWidget):
    fps_updated = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.chunk = VoxelChunk(32)
        self.cam_dist=50.; self.cam_rx=30.; self.cam_ry=45.
        self.cam_tgt=[16.,8.,16.]
        self._lp=None; self._mb=None
        self.wireframe=False; self.show_grid=True; self.show_axes=True
        self.show_bounds=True; self.auto_rotate=False
        self.fps=0.; self._fc=0; self._ft=time.time()
        self.setMinimumSize(400,300); self.setFocusPolicy(Qt.StrongFocus)
        self._rtimer=QTimer(self); self._rtimer.timeout.connect(self._autorot)
        self._rtimer.setInterval(16)

    def set_auto_rotate(self,on):
        self.auto_rotate=on
        if on: self._rtimer.start()
        else: self._rtimer.stop()

    def _autorot(self):
        self.cam_ry+=.3; self.update()

    def initializeGL(self):
        glClearColor(.10,.10,.14,1)
        glEnable(GL_DEPTH_TEST); glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LIGHTING); glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0,GL_POSITION,[1,2,1,0])
        glLightfv(GL_LIGHT0,GL_AMBIENT,[.30,.30,.30,1])
        glLightfv(GL_LIGHT0,GL_DIFFUSE,[.80,.80,.80,1])
        glLightfv(GL_LIGHT0,GL_SPECULAR,[.15,.15,.15,1])
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1,GL_POSITION,[-1,.5,-1,0])
        glLightfv(GL_LIGHT1,GL_AMBIENT,[0,0,0,1])
        glLightfv(GL_LIGHT1,GL_DIFFUSE,[.25,.25,.30,1])
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)

    def resizeGL(self,w,h):
        glViewport(0,0,w,h); glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(45,w/max(h,1),.1,500); glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClearColor(.10,.10,.14,1)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT); glLoadIdentity()
        rx,ry=math.radians(self.cam_rx),math.radians(self.cam_ry)
        ex=self.cam_tgt[0]+self.cam_dist*math.cos(rx)*math.sin(ry)
        ey=self.cam_tgt[1]+self.cam_dist*math.sin(rx)
        ez=self.cam_tgt[2]+self.cam_dist*math.cos(rx)*math.cos(ry)
        gluLookAt(ex,ey,ez,*self.cam_tgt,0,1,0)
        glLightfv(GL_LIGHT0,GL_POSITION,[ex+10,ey+20,ez+10,0])
        if self.show_grid: self._grid()
        if self.show_axes: self._axes()
        if self.show_bounds: self._bounds()
        self._voxels()
        self._fc+=1; now=time.time(); dt=now-self._ft
        if dt>=1.: self.fps=self._fc/dt; self._fc=0; self._ft=now; self.fps_updated.emit(self.fps)

    def _grid(self):
        glDisable(GL_LIGHTING); glColor4f(.30,.30,.35,.5); glLineWidth(1.); s=self.chunk.size
        glBegin(GL_LINES)
        for i in range(s+1): glVertex3f(i,0,0);glVertex3f(i,0,s);glVertex3f(0,0,i);glVertex3f(s,0,i)
        glEnd(); glEnable(GL_LIGHTING)

    def _axes(self):
        glDisable(GL_LIGHTING); glLineWidth(2.5); L=6.; glBegin(GL_LINES)
        glColor3f(1.,.25,.25); glVertex3f(0,0,0); glVertex3f(L,0,0)
        glColor3f(.25,1.,.25); glVertex3f(0,0,0); glVertex3f(0,L,0)
        glColor3f(.35,.35,1.); glVertex3f(0,0,0); glVertex3f(0,0,L)
        glEnd(); glLineWidth(1.); glEnable(GL_LIGHTING)

    def _bounds(self):
        glDisable(GL_LIGHTING); glColor4f(.50,.50,.70,.20); glLineWidth(1.); s=self.chunk.size
        glBegin(GL_LINES)
        for x in(0,s):
            for y in(0,s): glVertex3f(x,y,0);glVertex3f(x,y,s)
        for x in(0,s):
            for z in(0,s): glVertex3f(x,0,z);glVertex3f(x,s,z)
        for y in(0,s):
            for z in(0,s): glVertex3f(0,y,z);glVertex3f(s,y,z)
        glEnd(); glEnable(GL_LIGHTING)

    def _voxels(self):
        if self.chunk.mesh_dirty: self.chunk.build_mesh()
        if self.chunk.vertex_count==0: return
        if self.wireframe: glPolygonMode(GL_FRONT_AND_BACK,GL_LINE); glDisable(GL_LIGHTING)
        glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_NORMAL_ARRAY); glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3,GL_FLOAT,0,self.chunk.vertices)
        glNormalPointer(GL_FLOAT,0,self.chunk.normals)
        glColorPointer(4,GL_FLOAT,0,self.chunk.colors_arr)
        glDrawArrays(GL_TRIANGLES,0,self.chunk.vertex_count)
        glDisableClientState(GL_VERTEX_ARRAY); glDisableClientState(GL_NORMAL_ARRAY); glDisableClientState(GL_COLOR_ARRAY)
        if self.wireframe: glPolygonMode(GL_FRONT_AND_BACK,GL_FILL); glEnable(GL_LIGHTING)

    def mousePressEvent(self,e): self._lp=e.position();self._mb=e.button()
    def mouseReleaseEvent(self,e): self._lp=None;self._mb=None
    def mouseMoveEvent(self,e):
        if self._lp is None: return
        dx=e.position().x()-self._lp.x(); dy=e.position().y()-self._lp.y()
        if self._mb==Qt.LeftButton:
            self.cam_ry+=dx*.5;self.cam_rx=max(-89,min(89,self.cam_rx+dy*.5))
        elif self._mb==Qt.RightButton:
            self.cam_tgt[0]-=dx*.05;self.cam_tgt[1]+=dy*.05
        elif self._mb==Qt.MiddleButton:
            self.cam_dist=max(5,min(200,self.cam_dist+dy*.15))
        self._lp=e.position();self.update()
    def wheelEvent(self,e):
        self.cam_dist=max(5,min(200,self.cam_dist-e.angleDelta().y()/120*2));self.update()


# ================================================================
#  Dynamic Parameter Panel
# ================================================================
class ParamPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._lay=QVBoxLayout(self); self._lay.setContentsMargins(4,4,4,4)
        self._widgets: dict[str,QWidget]={}

    def clear(self):
        while self._lay.count():
            w=self._lay.takeAt(0).widget()
            if w: w.deleteLater()
        self._widgets.clear()

    def build_from(self, params):
        self.clear()
        for key, pd in params.items():
            row=QHBoxLayout()
            lbl=QLabel(pd["label"]+":"); lbl.setMinimumWidth(85); lbl.setToolTip(pd.get("tooltip",""))
            row.addWidget(lbl)
            if pd["type"]==int:
                w=QSpinBox(); w.setRange(pd["min"],pd["max"]); w.setSingleStep(pd.get("step",1)); w.setValue(pd["default"])
            elif pd["type"]==float:
                w=QDoubleSpinBox(); w.setRange(pd["min"],pd["max"]); w.setSingleStep(pd.get("step",.01))
                w.setDecimals(3); w.setValue(pd["default"])
            elif pd["type"]=="combo":
                w=QComboBox()
                for val,name in pd.get("options",{}).items(): w.addItem(name,val)
            else:
                w=QLabel(str(pd["default"]))
            w.setToolTip(pd.get("tooltip","")); w.setMinimumWidth(75)
            row.addWidget(w); self._lay.addLayout(row); self._widgets[key]=w

    def get_values(self):
        v={}
        for k,w in self._widgets.items():
            if isinstance(w,QSpinBox): v[k]=w.value()
            elif isinstance(w,QDoubleSpinBox): v[k]=w.value()
            elif isinstance(w,QComboBox): v[k]=w.currentData()
        return v


# ================================================================
#  Main Window
# ================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voxel Engine Testing Tool v2.0")
        self.resize(1400,900)
        self.algos=_reg_algos(); self.ops=_reg_ops(); self.undo_stack=UndoStack()
        self._build_toolbar(); self._build_central()
        self.status_bar=QStatusBar(); self.setStatusBar(self.status_bar)
        self.status_lbl=QLabel("LMB=Orbit | RMB=Pan | Wheel=Zoom | Space=Auto-rotate | G=Generate | Ctrl+Z/Y=Undo/Redo")
        self.status_bar.addPermanentWidget(self.status_lbl)
        self._stimer=QTimer(self);self._stimer.timeout.connect(self._update_stats);self._stimer.start(300)
        self._build_shortcuts()
        self._gen_t=0.;self._mesh_t=0.
        QTimer.singleShot(200,self._on_generate)

    def _build_toolbar(self):
        tb=QToolBar("Main");tb.setMovable(False);tb.setIconSize(QSize(20,20));self.addToolBar(tb)
        for name,tip,fn in [
            ("Generate","Generate (G)",self._on_generate),
            ("Undo","Undo (Ctrl+Z)",self._on_undo),
            ("Redo","Redo (Ctrl+Y)",self._on_redo),
        ]:
            a=QAction(name,self);a.setToolTip(tip);a.triggered.connect(fn);tb.addAction(a)
        tb.addSeparator()
        for name,tip,fn in [
            ("Screenshot","Save PNG (Ctrl+S)",self._on_screenshot),
            ("Clear","Clear all voxels",self._on_clear),
            ("Benchmark","Run benchmark",self._on_benchmark),
        ]:
            a=QAction(name,self);a.setToolTip(tip);a.triggered.connect(fn);tb.addAction(a)

    def _build_central(self):
        sp=QSplitter(Qt.Horizontal);self.setCentralWidget(sp)
        self.viewport=VoxelViewport();self.viewport.fps_updated.connect(lambda _:None)
        sp.addWidget(self.viewport)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setMinimumWidth(310);scroll.setMaximumWidth(400)
        panel=QWidget();self._play=QVBoxLayout(panel);self._play.setSpacing(8)
        scroll.setWidget(panel);sp.addWidget(scroll);sp.setStretchFactor(0,4);sp.setStretchFactor(1,0)
        self._build_gen_group();self._build_ops_group();self._build_csg_group()
        self._build_viz_group();self._build_cam_group();self._build_stats_group()
        self._play.addStretch()

    def _build_gen_group(self):
        gb=QGroupBox("Generation Algorithms");gl=QVBoxLayout(gb);gl.setSpacing(6)
        self.gen_combo=QComboBox();cats=sorted(set(a.category for a in self.algos.values()))
        for cat in cats:
            self.gen_combo.addItem(f"-- {cat} --")
            for name,info in self.algos.items():
                if info.category==cat:
                    self.gen_combo.addItem(f"  {name}")
                    self.gen_combo.setItemData(self.gen_combo.count()-1,name,Qt.UserRole)
        gl.addWidget(self.gen_combo)
        row=QHBoxLayout();row.addWidget(QLabel("Seed:"))
        self.seed_spin=QSpinBox();self.seed_spin.setRange(0,99999);self.seed_spin.setValue(42)
        self.seed_spin.setToolTip("Random seed");row.addWidget(self.seed_spin)
        row.addWidget(QLabel("Size:"));self.size_combo=QComboBox()
        self.size_combo.addItems(["16","32","48","64"]);self.size_combo.setCurrentIndex(1)
        self.size_combo.setToolTip("Chunk NxNxN");row.addWidget(self.size_combo);gl.addLayout(row)
        self.gen_pp=ParamPanel();gl.addWidget(self.gen_pp)
        self.gen_combo.currentIndexChanged.connect(self._on_gen_changed);self._on_gen_changed()
        btn=QPushButton("Generate");btn.setObjectName("generateBtn")
        btn.setToolTip("Generate selected algorithm");btn.clicked.connect(self._on_generate)
        gl.addWidget(btn);self._play.addWidget(gb)

    def _build_ops_group(self):
        gb=QGroupBox("Voxel Operations");ol=QVBoxLayout(gb);ol.setSpacing(6)
        self.ops_combo=QComboBox()
        for name in self.ops: self.ops_combo.addItem(name)
        ol.addWidget(self.ops_combo)
        self.ops_pp=ParamPanel();ol.addWidget(self.ops_pp)
        self.ops_combo.currentIndexChanged.connect(self._on_op_changed);self._on_op_changed()
        btn=QPushButton("Apply Operation");btn.setObjectName("opsBtn")
        btn.setToolTip("Apply selected operation");btn.clicked.connect(self._on_operation)
        ol.addWidget(btn);self._play.addWidget(gb)

    def _build_csg_group(self):
        gb=QGroupBox("CSG Boolean");cl=QVBoxLayout(gb);cl.setSpacing(6)
        cl.addWidget(QLabel("Combine with shape:"))
        self.csg_shape=QComboBox();self.csg_shape.addItems(["Sphere","Cube","Cylinder"])
        self.csg_shape.setToolTip("Shape for CSG");cl.addWidget(self.csg_shape)
        row=QHBoxLayout()
        for name,op in [("Union","union"),("Intersect","intersect"),("Subtract","diff")]:
            b=QPushButton(name);b.setToolTip(f"CSG {name}")
            b.clicked.connect(lambda _,o=op:self._on_csg(o));row.addWidget(b)
        cl.addLayout(row);self._play.addWidget(gb)

    def _build_viz_group(self):
        gb=QGroupBox("Visualization");vl=QVBoxLayout(gb);vl.setSpacing(4)
        for label,attr,df,tip in [
            ("Wireframe","wireframe",False,"Wireframe render"),
            ("Show Grid","show_grid",True,"Ground grid"),
            ("Show Axes","show_axes",True,"XYZ axes"),
            ("Show Bounds","show_bounds",True,"Bounding box"),
            ("Auto Rotate","auto_rotate",False,"Continuous rotation (Space)"),
        ]:
            cb=QCheckBox(label);cb.setChecked(default=df);cb.setToolTip(tip)
            if attr=="auto_rotate": cb.toggled.connect(self._on_ar_toggle)
            else: cb.toggled.connect(lambda v,a=attr:(setattr(self.viewport,a,v),self.viewport.update()))
            vl.addWidget(cb)
        self._play.addWidget(gb)

    def _build_cam_group(self):
        gb=QGroupBox("Camera");cl=QVBoxLayout(gb);cl.setSpacing(4)
        r1=QHBoxLayout();r2=QHBoxLayout()
        for i,(label,rx,ry,tip) in enumerate([
            ("Reset",30,45,"Default view"),("Top",89,0,"Top-down"),
            ("Front",0,0,"Front view"),("Side",0,90,"Side view"),("Iso",30,45,"Isometric"),
        ]):
            b=QPushButton(label);b.setToolTip(tip)
            b.clicked.connect(lambda _,x=rx,y=ry:self._set_cam(x,y))
            (r1 if i<3 else r2).addWidget(b)
        cl.addLayout(r1);cl.addLayout(r2);self._play.addWidget(gb)

    def _build_stats_group(self):
        gb=QGroupBox("Statistics");sl=QVBoxLayout(gb)
        self.stats_lbl=QLabel("Voxels:   --\nFaces:    --\nVertices: --\nFPS:      --\nGen Time: --\nMesh Time: --")
        self.stats_lbl.setObjectName("statsLabel");sl.addWidget(self.stats_lbl)
        self._play.addWidget(gb)

    def _build_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Z"),self,self._on_undo)
        QShortcut(QKeySequence("Ctrl+Y"),self,self._on_redo)
        QShortcut(QKeySequence("Ctrl+S"),self,self._on_screenshot)
        QShortcut(QKeySequence("G"),self,self._on_generate)
        QShortcut(QKeySequence("F"),self,self._toggle_panel)
        QShortcut(QKeySequence("Space"),self,self._toggle_ar)

    # ---- Slots ----
    def _on_gen_changed(self):
        name=self.gen_combo.currentData(Qt.UserRole)
        if name and name in self.algos: self.gen_pp.build_from(self.algos[name].params)
        else: self.gen_pp.clear()

    def _on_op_changed(self):
        n=self.ops_combo.currentText()
        if n in self.ops: self.ops_pp.build_from(self.ops[n].params)
        else: self.ops_pp.clear()

    def _on_generate(self):
        name=self.gen_combo.currentData(Qt.UserRole)
        if not name or name not in self.algos: return
        self.undo_stack.push(self.viewport.chunk)
        sz=self._csz(); c=VoxelChunk(sz); seed=self.seed_spin.value()
        params=self.gen_pp.get_values()
        t=time.time();self.algos[name].fn(c,seed,params);gt=(time.time()-t)*1000
        t=time.time();c.build_mesh();mt=(time.time()-t)*1000
        self._gen_t=gt;self._mesh_t=mt;self.viewport.chunk=c;self._set_cam(30,45)
        self.viewport.update()
        self.status_lbl.setText(f"'{name}' -- gen: {gt:.1f}ms, mesh: {mt:.1f}ms ({sz}x{sz}x{sz})")

    def _on_operation(self):
        on=self.ops_combo.currentText()
        if on not in self.ops: return
        self.undo_stack.push(self.viewport.chunk);c=self.viewport.chunk
        seed=self.seed_spin.value();params=self.ops_pp.get_values()
        t=time.time();self.ops[on].fn(c,seed,params);mt=(time.time()-t)*1000
        self._mesh_t=mt
        if on=="Subsample (2x)" and c.size!=self._csz(): self.viewport.chunk=c
        self.viewport.update();self.status_lbl.setText(f"'{on}' in {mt:.1f}ms")

    def _on_csg(self,op):
        self.undo_stack.push(self.viewport.chunk);c=self.viewport.chunk
        b=_make_shape(c.size,self.csg_shape.currentText());t=time.time()
        if op=="union": r=_csg_union(c,b)
        elif op=="intersect": r=_csg_isect(c,b)
        elif op=="diff": r=_csg_diff(c,b)
        else: return
        ms=(time.time()-t)*1000;self.viewport.chunk=r;self.viewport.update()
        self.status_lbl.setText(f"CSG {op} in {ms:.1f}ms")

    def _on_undo(self):
        r=self.undo_stack.undo(self.viewport.chunk)
        if r: self.viewport.chunk=r;self.viewport.update();self.status_lbl.setText("Undo")
        else: self.status_lbl.setText("Nothing to undo")

    def _on_redo(self):
        r=self.undo_stack.redo(self.viewport.chunk)
        if r: self.viewport.chunk=r;self.viewport.update();self.status_lbl.setText("Redo")
        else: self.status_lbl.setText("Nothing to redo")

    def _on_clear(self):
        self.undo_stack.push(self.viewport.chunk)
        self.viewport.chunk=VoxelChunk(self._csz());self.viewport.update()
        self.status_lbl.setText("Cleared")

    def _on_screenshot(self):
        path,_=QFileDialog.getSaveFileName(self,"Save Screenshot","voxel_screenshot.png","PNG (*.png);;All (*)")
        if path: self.viewport.grabFramebuffer().save(path);self.status_lbl.setText(f"Saved: {path}")

    def _on_benchmark(self):
        lines=[]
        for sz in(16,32):
            c=VoxelChunk(sz);t=time.time()
            self.algos["Perlin Heightmap"].fn(c,42,{"height_scale":sz//3,"noise_scale":.05});g=(time.time()-t)*1000
            t=time.time();c.build_mesh();m=(time.time()-t)*1000
            lines.append(f"Terrain {sz}x{sz}x{sz}:\n  Gen:  {g:>8.1f} ms\n  Mesh: {m:>8.1f} ms\n  Vox:  {c.count_voxels():>8,}\n  Tri:  {c.face_count:>8,}")
        c=VoxelChunk(16);t=time.time()
        self.algos["3D Perlin Noise"].fn(c,42,{"threshold":0.,"noise_scale":.08});g=(time.time()-t)*1000
        t=time.time();c.build_mesh();m=(time.time()-t)*1000
        lines.append(f"\n3D Noise 16x16x16:\n  Gen:  {g:>8.1f} ms\n  Mesh: {m:>8.1f} ms\n  Vox:  {c.count_voxels():>8,}")
        QMessageBox.information(self,"Benchmark","\n".join(lines))

    def _on_ar_toggle(self,on): self.viewport.set_auto_rotate(on)
    def _toggle_ar(self): self.viewport.set_auto_rotate(not self.viewport.auto_rotate)
    def _toggle_panel(self):
        sp=self.centralWidget()
        if sp.count()>1: sp.widget(1).setVisible(not sp.widget(1).isVisible())

    def _csz(self): return int(self.size_combo.currentText())
    def _set_cam(self,rx,ry):
        s=self.viewport.chunk.size
        self.viewport.cam_rx=rx;self.viewport.cam_ry=ry
        self.viewport.cam_tgt=[s/2.,s/4.,s/2.];self.viewport.cam_dist=s*1.6;self.viewport.update()

    def _update_stats(self):
        c=self.viewport.chunk;vc=c.count_voxels();bt=c.count_by_type()
        tl=["  {0}: {1:,}".format(BLOCK_NAMES.get(b,f"Type{b}"),n) for b,n in sorted(bt.items(),key=lambda x:-x[1])]
        ts="\n".join(tl) if tl else "  (empty)"
        self.stats_lbl.setText(
            f"Voxels:    {vc:>8,}\nFaces:     {c.face_count:>8,}\n"
            f"Vertices:  {c.vertex_count:>8,}\nFPS:       {self.viewport.fps:>8.1f}\n"
            f"Gen Time:  {self._gen_t:>7.1f} ms\nMesh Time: {self._mesh_t:>7.1f} ms\n\nBy Type:\n{ts}")


# ================================================================
#  Dark Theme QSS (Catppuccin Mocha inspired)
# ================================================================
DARK_QSS="""
QMainWindow, QWidget {
    background-color: #1e1e2e; color: #cdd6f4;
    font-family: "Segoe UI","Noto Sans",sans-serif; font-size: 13px;
}
QGroupBox {
    border: 1px solid #45475a; border-radius: 6px;
    margin-top: 14px; padding: 14px 8px 8px 8px;
    font-weight: bold; color: #cdd6f4;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    padding: 2px 10px; color: #89b4fa; background-color: #1e1e2e; border-radius: 4px;
}
QPushButton {
    background-color: #313244; border: 1px solid #45475a; border-radius: 5px;
    padding: 6px 14px; color: #cdd6f4; font-weight: bold; min-height: 22px;
}
QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }
QPushButton:pressed { background-color: #585b70; }
QPushButton#generateBtn { background-color: #2d5a2d; border-color: #40a02b; color: #a6e3a1; }
QPushButton#generateBtn:hover { background-color: #3a7040; }
QPushButton#opsBtn { background-color: #2a4a6b; border-color: #89b4fa; color: #89dceb; }
QPushButton#opsBtn:hover { background-color: #345a80; }
QComboBox {
    background-color: #313244; border: 1px solid #45475a; border-radius: 5px;
    padding: 4px 8px; color: #cdd6f4; min-height: 22px;
}
QComboBox:hover { border-color: #89b4fa; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background-color: #313244; color: #cdd6f4; border: 1px solid #45475a;
    selection-background-color: #45475a; selection-color: #cdd6f4; outline: none;
}
QSpinBox, QDoubleSpinBox {
    background-color: #313244; border: 1px solid #45475a; border-radius: 5px;
    padding: 3px 8px; color: #cdd6f4; min-height: 22px;
}
QSpinBox:hover, QDoubleSpinBox:hover { border-color: #89b4fa; }
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: #45475a; border: none; width: 18px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #585b70;
}
QCheckBox { color: #cdd6f4; spacing: 8px; }
QCheckBox::indicator {
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid #45475a; background-color: #313244;
}
QCheckBox::indicator:checked { background-color: #89b4fa; border-color: #89b4fa; }
QCheckBox::indicator:hover { border-color: #89b4fa; }
QLabel { color: #bac2de; background: transparent; }
QLabel#statsLabel {
    font-family: "JetBrains Mono","Fira Code","Consolas",monospace;
    font-size: 11px; color: #a6adc8; padding: 4px;
}
QScrollArea { border: none; background-color: #181825; }
QScrollBar:vertical { background-color: #181825; width: 10px; border: none; }
QScrollBar::handle:vertical { background-color: #45475a; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background-color: #585b70; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background-color: #181825; height: 10px; border: none; }
QScrollBar::handle:horizontal { background-color: #45475a; border-radius: 5px; min-width: 30px; }
QToolBar {
    background-color: #181825; border-bottom: 1px solid #313244; padding: 4px; spacing: 6px;
}
QToolBar QToolButton {
    color: #cdd6f4; background: transparent; border: 1px solid transparent;
    border-radius: 4px; padding: 5px 12px; font-weight: bold;
}
QToolBar QToolButton:hover { background-color: #313244; border-color: #45475a; }
QToolBar QToolButton:pressed { background-color: #45475a; }
QStatusBar {
    background-color: #181825; border-top: 1px solid #313244;
    color: #a6adc8; font-size: 12px; padding: 2px 8px;
}
QSplitter::handle { background-color: #313244; width: 2px; }
QMessageBox { background-color: #1e1e2e; }
QMessageBox QLabel { color: #cdd6f4; font-size: 13px; }
QMessageBox QPushButton { min-width: 80px; }
"""


# ================================================================
#  Entry Point
# ================================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()