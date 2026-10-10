from pathlib import Path
import json,math
import numpy as np

def compare(nu,n):
 u=np.linspace(-12,12,n); eta=math.sqrt(12/nu)*u
 small=abs(eta)<.02; c=np.empty_like(eta); z=eta[small]**2
 c[small]=z*(1/24+z*(-1/2880+z*(1/181440-z/9676800)))
 h=eta[~small]/2; c[~small]=np.log(np.sinh(h)/h)
 p=np.exp(-nu*c);p/=np.trapezoid(p,u)
 q=np.exp(-u*u/2)/math.sqrt(2*math.pi)
 cdf=np.r_[0,np.cumsum(.5*(p[:-1]+p[1:])*np.diff(u))]
 normal=np.array([.5*math.erfc(-float(v)/math.sqrt(2)) for v in u])
 return dict(nu=nu,n=n,ks=float(max(abs(cdf-normal))),
             hellinger_squared=float(.5*np.trapezoid((np.sqrt(p)-np.sqrt(q))**2,u)),
             standardized_variance=float(np.trapezoid(u*u*p,u)))
rows=[compare(nu,n) for nu in [200,500,1000,5000] for n in [65537,131073]]
print(json.dumps(rows,indent=2));Path(__file__).with_suffix('.json').write_text(json.dumps(rows,indent=2)+'\n')
