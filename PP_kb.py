#import wget
import h5py
import numpy as np
import os
import re

from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader

rep=''
npart=12
order=5


nx=62
ny=19
nz=60


#pip install boto3
ACCESS_KEY = 'AKIAYALVQGAYMYX7ZR7L'
SECRET_KEY = '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'
bucket_name = 'kbwiki-data'
from boto3.session import Session
import boto3


session = Session(aws_access_key_id=ACCESS_KEY,
              aws_secret_access_key=SECRET_KEY)
s3 = session.resource('s3')
bucket = s3.Bucket(bucket_name)
#%%


#################     Build mesh
def extract_grid(npart=npart,nx=nx,ny=ny,nz=nz,order=order):
    filename='Channel-180.pyfrm'
   # url='http://kbwiki-data.s3-eu-west-2.amazonaws.com/index.html?prefix=DNS-1/2/Channel_180/setup/Channel-180.pyfrm'
    #wget.download(url,out=filename) 
    bucket.download_file('DNS-1/2/Channel_180/setup/Channel-180.pyfrm','Channel-180.pyfrm')
    print(filename)
    re = h5py.File(filename, 'r')
    grid=[]
    for i in range(npart):
        part=f'spt_hex_p{i}'
        tmp=re[part]
        gtmp=tmp[()]
        if i==0:
            grid= gtmp
        else:
            grid=np.concatenate( (grid, gtmp),1)
    
   

    pnty=np.squeeze(grid[0,:,1])
    pnty=np.expand_dims(pnty,1)
    indi=np.arange(0,len(pnty))
    indi=np.expand_dims(indi,1)
    sup=np.concatenate((pnty,indi),1)

    supt=sup[sup[:,0].argsort()]

    for i in range(ny):
        bd=np.arange(i*nx*nz,(i+1)*nx*nz)
        tmp=supt[bd,1]
        tmp=np.expand_dims(tmp,1)
        if i==0:
            sliceY= tmp
        else:
            sliceY=np.concatenate( (sliceY, tmp),1)
     
    Mesh= np.zeros((nx,nz,ny))
    for y in range(ny):
        yi=np.asarray(np.squeeze(sliceY[:,y]), dtype=int)
        pntx=np.squeeze(grid[0,yi,0])       
        pntx=np.expand_dims(pntx,1)
        indi=np.arange(0,len(pntx))
        indi=np.expand_dims(indi,1)
        sup=np.concatenate((pntx,indi),1)       
        supx=sup[sup[:,0].argsort()]
        supx=np.reshape(supx,(nx,nz,2))
    
        for x in range(nx):
            tmp=np.asarray(supx[x,:,1], dtype=int)   
            xi=yi[tmp]
            pntz=np.squeeze(grid[0,xi,2])       
            pntz=np.expand_dims(pntz,1)
            indi=np.arange(0,len(pntz))
            indi=np.expand_dims(indi,1)
            sup=np.concatenate((pntz,indi),1)  
            supz=sup[sup[:,0].argsort()]
            tmp=np.asarray(supz[:,1], dtype=int) 
            zi=xi[tmp]
            Mesh[x,:,y]=zi
    Mesh=np.asarray(Mesh, dtype=int)       
    mesh=np.reshape(Mesh,(nx*nz*ny))        
    grid=grid[:,mesh,:]
    #grid=np.reshape(grid,(8,nx,nz,ny,3))
    grid=np.reshape(grid,(8,ny,nz,nx,3),order='F')
    
    n=f'PyFR/pyfr/quadrules/hex/gauss-legendre-n{(order+1)**3}-d{2*(order+1)-1}-spu.txt'
    #n=f'/home/cfd/anaconda3/lib/python3.7/site-packages/pyfr/quadrules/hex/gauss-legendre-n{(order+1)**3}-d{2*(order+1)-1}-spu.txt'

    f=open(n, 'r')
    k=f.read()
    k=k.replace("\n", " ")

    k = map(float, k.split())
    k=list(k)
    k=np.asarray(k)
    k=np.reshape(k,((order+1)**3,4))
    
    kk=np.reshape(k,(order+1,order+1,order+1,4))

    dx=kk[0,0,:,0]
    dy=kk[0,:,0,1]
    dz=kk[:,0,0,2]

    dx=(dx+1)/2
    Lx=[]
    for i in range(nx):
        Cx=grid[(0,4),0,0,i,0]
        d=dx*(Cx[1]-Cx[0])+Cx[0]
        Lx=np.append(Lx,d)

    dy=(dy+1)/2
    Ly=[]
    for i in range(ny):
        Cy=grid[(0,2),i,0,0,1]
        d=dy*(Cy[1]-Cy[0])+Cy[0]
        Ly=np.append(Ly,d)  
    
    dz=(dz+1)/2
    
    Lz=[]
    for i in range(nz):
        Cz=grid[(0,1),0,i,0,2]
        d=dz*(Cz[1]-Cz[0])+Cz[0]
        Lz=np.append(Lz,np.flipud(d))   
    os.remove(filename) 
    return mesh, Lx,Ly,Lz




#%%  load snaps from dropbox
def download_snap(time_step):

    url=f'DNS-1/2/Channel_180/snapshots/Channel-{time_step:010.4f}.pyfrs'
    bucket.download_file(url,f'Channel-{time_step:010.4f}.pyfrs')
    print(url)

    return url

#%% load stats from dropbox
def download_stats(Var):
  
    url=f'DNS-1/2/Channel_180/statistics/avg-' + Var + '.pyfrs' 
    bucket.download_file(url,f'avg-' + Var + '.pyfrs')
 

    
    print(url)
    return url


#%%

def build_fields(sol,K):
    u=sol[:,int(K),:,:,:]
    n,ny,nz,nx=u.shape[:]
    n=int(np.asarray( np.cbrt(n), dtype=int))   
    u=np.reshape(u,(n,n**2,ny,nz,nx),order='F')
    u=np.transpose(u,(1,0,2,3,4))
    u=np.reshape(u,(n,n,n,ny,nz,nx),order='F')
    u=np.transpose(u,(0,3,1,2,4,5))
    u=np.reshape(u,(n*ny,n,n,nz,nx),order='F')
    u=np.squeeze(u[:,:,:,::-1,:])
    u=np.reshape(u,(n*ny,n,nz*n,nx),order='F')
    u=np.transpose(u,(0,2,1,3))
    u=np.reshape(u,(n*ny,nz*n,nx*n),order='F')
    u=np.transpose(u,(2,0,1))
    return u
################### build field ################################

Mesh,Lx,Ly,Lz =extract_grid()



def Read_solutions(time_step,Mesh=Mesh):
    #Mesh,Lx,Ly,Lz =extract_grid()
    nx=62
    ny=19
    nz=60
    
    filename = f'Channel-{time_step:010.4f}.pyfrs'
    
    url=download_snap(time_step)
    wget.download(url,out=filename)
    
    soln = NativeReader(filename)
    cfg=Inifile(soln['stats'])
    vari=cfg.get('data','fields')
    vari = [s.strip() for s in vari.split(',')]
    
    re = h5py.File(filename, 'r')
    sol=[]
    for i in range(npart):
        part=f'soln_hex_p{i}'
        tmp=re[part]
        gtmp=tmp[()]
        if i==0:
            sol= gtmp
        else:
            sol=np.concatenate( (sol, gtmp),2)
    
    sol=sol[:,:,Mesh]
    nk,nv,_=sol.shape[:]
    sol=np.reshape(sol,(nk,nv,ny,nz,nx),order='F')
    
    

    
    rho=build_fields(sol,0)
    rhou=build_fields(sol,1)
    rhov=build_fields(sol,2)
    rhow=build_fields(sol,3)
    E=build_fields(sol,4)
    
    os.remove(filename)
    return rho,rhou,rhov,rhow,E



def Read_stats(vari,Mesh=Mesh):
    
    url=download_stats(vari) 
    filename='avg-' + vari +'.pyfrs'
    wget.download(url,out=filename)
    
    
    nx=62
    ny=19
    nz=60
    
    soln = NativeReader(filename)
    #cfg=Inifile(soln['stats'])
    #vari=cfg.get('data','fields')
    #vari = [s.strip() for s in vari.split(',')]
    
    
    dre = h5py.File(filename, 'r')
    sol=[]
    for i in range(npart):
        part=f'tavg_hex_p{i}'
        tmp=dre[part]
        gtmp=tmp[()]
        if i==0:
            sol= gtmp
        else:
            sol=np.concatenate( (sol, gtmp),2)
    
    sol=sol[:,:,Mesh]
    nk,nv,_=sol.shape[:]
    sol=np.reshape(sol,(nk,nv,ny,nz,nx),order='F')
    
    u=build_fields(sol,0)
    
    return u


def wallunit(A,utau,normi,sym=1):
    Ly=ny*(order+1)
    A=A[:int(1+np.floor(Ly/2))]+sym*np.flipud(A[int(np.floor(Ly/2)-1):Ly])
    A=0.5*A/(utau**normi)
    return A

def MeanXZ(u,utau,normi,sym=1):    
    #u=Fields(sol)
    Um=np.average(u,(0,2))
    Up=wallunit(Um,utau,normi,sym)
    return Um, Up


def MeanX(u,utau,normi,sym=1):    
    #u=Fields(sol)
    Um=np.average(u,(0))
    Up=wallunit(Um,utau,normi,sym)
    return Um, Up

def MeanZ(u,utau,normi,sym=1):    
    #u=Fields(sol)
    Um=np.average(u,(2))
    Up=wallunit(Um.T,utau,normi,sym)
    return Um, Up




