#!/usr/bin/env python3
"""Independent exact-CF envelope, population-density and stream qualification.

Requires NumPy and a C compiler; does not use the sampler's B implementation
for the reference. An 80-digit reference checks the curvature/tangent bounds.
The accepted mixture density is reconstructed independently from component
weights: this detects the symmetric normal bias without a 100M Monte Carlo run.
"""
import argparse
import ctypes as C
import decimal as D
import json
import math
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
P=C.POINTER(C.c_double)

def library(work,flags=()):
    work.mkdir(parents=True,exist_ok=True)
    probe=work/'probe.c'
    probe.write_text((ROOT/'tools/exact_cf/probe.c').read_text().replace('CORE_FILE',str(ROOT/'src/c/cb_core.c')))
    so=work/'cf.so'
    subprocess.run([os.environ.get('CC','clang'),'-O3','-shared','-fPIC',
                    '-DCB_VALIDATE_ETA','-DCB_VALIDATE_CF',*flags,
                    '-I',str(ROOT/'src/c'),str(probe),'-lm','-o',str(so)],check=True,capture_output=True)
    lib=C.CDLL(str(so))
    lib.cf_state.argtypes=[C.c_double,C.c_double,P]
    lib.cf_samples.argtypes=[C.c_double,C.c_double,C.c_int,P,C.c_int,P]
    lib.cf_relative.argtypes=[C.c_double,C.c_double,C.c_int,P,P]
    return lib

def log_relative(eta,mode,chi,nu):
    # Long-double hyperbolic identity, independent of the C rational kernel.
    x=np.asarray(eta,dtype=np.longdouble)
    def centered(a):
        a=np.asarray(a,dtype=np.longdouble);z=a*a
        small=abs(a)<.02
        with np.errstate(invalid='ignore',divide='ignore'):
            direct=np.log(np.sinh(a/2)/(a/2))
        return np.where(small,z*(1/24+z*(-1/2880+z*(1/181440-z/9676800))),direct)
    return np.asarray((np.longdouble(chi)-np.longdouble(nu)/2)*(x-mode)
                      -nu*(centered(x)-centered(mode)),dtype=float)

def exact_f(chi,nu,eta):
    c,n,x=map(D.Decimal,(chi,nu,eta))
    if not x: return D.Decimal(0)
    return c*x-n*((x.exp()-1)/x).ln()

def audit(lib,v,chi,nu):
    mode,sigma,r,kmin,kmax,d,shift,sd,pad,lf,rf,ls,rs,cm,lm=v[1:]
    rate=min(chi,nu-chi)
    with D.localcontext() as ctx:
        ctx.prec=80
        peak=exact_f(rate,nu,mode)
        for delta in sorted(set(np.r_[np.linspace(-40*sigma,40*sigma,81),-r,r,0.0])):
            eta=mode+float(delta);delta=eta-mode
            f=float(exact_f(rate,nu,eta)-peak)
            if abs(delta)<=r:
                env=d*delta-.5*kmin*delta*delta+pad
                lower=d*delta-.5*kmax*delta*delta-pad
                assert lower<=f<=env,(chi,nu,eta,lower,f,env)
                assert 1-(.5*(kmax-kmin)*delta*delta+2*pad)<=math.exp(f-env)
            else:
                env=lf+ls*(delta+r)+pad if delta<0 else rf+rs*(delta-r)+pad
                assert f<=env,(chi,nu,eta,f,env)
    grid=mode+sigma*np.linspace(-40,40,131073);delta=grid-mode
    f=log_relative(grid,mode,rate,nu)
    center=abs(delta)<=r;left=delta<-r
    env=np.where(center,d*delta-.5*kmin*delta**2+pad,
                 np.where(left,lf+ls*(delta+r)+pad,rf+rs*(delta-r)+pad))
    logq=np.where(center,math.log(cm)-math.log(sd)-.5*math.log(2*math.pi)-.5*((delta-shift)/sd)**2,
                  np.where(left,math.log(lm*ls)+ls*(delta+r),math.log((1-cm-lm)*(-rs))+rs*(delta-r)))
    # Accepted density of the actual component sampler and full-target test.
    full=np.empty_like(grid)
    assert lib.cf_relative(chi,nu,len(grid),grid.ctypes.data_as(P),full.ctypes.data_as(P))
    acceptance=np.exp(full-env)
    cheap=np.maximum(0.0,1.0-(.5*(kmax-kmin)*delta**2+2*pad))
    assert np.all(cheap[center]<=acceptance[center])
    # Actual union of squeeze and full-density decisions with the same U.
    acceptance[center]=np.maximum(cheap[center],acceptance[center])
    corrected=np.exp(logq)*acceptance;corrected/=np.trapezoid(corrected,grid)
    reference=np.exp(f);mass=np.trapezoid(reference,grid);reference/=mass
    h2=float(.5*np.trapezoid((np.sqrt(corrected)-np.sqrt(reference))**2,grid))
    assert h2<1e-15,(chi,nu,h2)
    masses=.5*(reference[:-1]+reference[1:])*np.diff(grid)
    cdf=np.r_[0,np.cumsum(masses)];cdf/=cdf[-1]
    coarse=np.r_[0,np.cumsum(.5*(reference[::2][:-1]+reference[::2][1:])*np.diff(grid[::2]))]
    coarse/=coarse[-1]
    reference_error=float(max(abs(cdf-np.interp(grid,grid[::2],coarse))))
    assert reference_error<5e-7
    with D.localcontext() as ctx:
        ctx.prec=80;peak=exact_f(rate,nu,mode);bounds=[]
        for edge,sign in [(grid[0],1),(grid[-1],-1)]:
            x=D.Decimal(float(edge));e=x.exp()
            slope=D.Decimal(rate)-D.Decimal(nu)*(e/(e-1)-1/x)
            assert sign*slope>0
            bounds.append(float(exact_f(rate,nu,float(edge))-peak)-math.log(abs(float(slope)))-math.log(float(mass)))
        log_tail_bound=max(bounds)+math.log(2)
        assert log_tail_bound<math.log(1e-10)
    return grid,cdf,h2,reference_error,log_tail_bound

def run(work,n=100000):
    lib=library(work);rows=[]
    states=[(nu/2,nu) for nu in (301.,500.,1000.,5000.,1e6)]
    states += [(249.5,500.),(250.5,500.),(10000.,1e6),(990000.,1e6)]
    for a in (1-1e-6,1+1e-6):
        rate=1/a-1/math.expm1(a)
        states.extend([(rate*1e6,1e6),((1-rate)*1e6,1e6)])
    for chi,nu in states:
        v=(C.c_double*16)();assert lib.cf_state(chi,nu,v)
        grid,cdf,h2,ref_error,tail_bound=audit(lib,list(v),chi,nu)
        x=np.empty(n);stats=(C.c_double*5)()
        assert lib.cf_samples(chi,nu,n,x.ctypes.data_as(P),0,stats)
        if chi>nu/2:x=-x
        pit=np.sort(np.interp(x,grid,cdf));ranks=np.arange(1,n+1)
        ks=float(max(np.max(ranks/n-pit),np.max(pit-(ranks-1)/n)))
        assert ks<=3/math.sqrt(n)+5e-7,(chi,nu,ks)
        rows.append(dict(chi=chi,nu=nu,n=n,ks=ks,accepted_density_h2=h2,reference_error=ref_error,log_reference_tail_bound=tail_bound,
                         proposals=float(stats[0]),evaluations=float(stats[1]),
                         squeezes=float(stats[2]),tails=float(stats[3])))
    # Single buffered and direct large-batch reflection use the same RNG path.
    for path in (0,1):
        a=np.empty(10000);b=np.empty_like(a);v=(C.c_double*5)()
        assert lib.cf_samples(249.5,500.,len(a),a.ctypes.data_as(P),path,v)
        assert lib.cf_samples(250.5,500.,len(b),b.ctypes.data_as(P),path,v)
        assert np.array_equal(a,-b)
    # Exercise a rare exponential component, not just the Gaussian center.
    x=np.empty(2000000);v=(C.c_double*5)()
    assert lib.cf_samples(250.,500.,len(x),x.ctypes.data_as(P),0,v)
    assert v[3]>0 and v[1]>0 and v[2]>0
    state=(C.c_double*16)();assert lib.cf_state(250.,500.,state)
    assert np.any(abs(x)>state[3]), "No accepted exponential-tail samples"
    # Both public draw paths propagate a finite rejection-budget failure.
    for nfail in (1,65):
        x=np.empty(nfail)
        assert not lib.cf_samples(250.,500.,nfail,x.ctypes.data_as(P),3,v)
    (work/'qualification.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--n',type=int,default=100000)
    a=p.parse_args();print(json.dumps(run(a.output.resolve(),a.n),indent=2))
