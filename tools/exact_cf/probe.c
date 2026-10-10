/* Private exact CF qualification and counter-free benchmark harness. */
#include <time.h>
#include "CORE_FILE"
static double seconds(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int cf_state(double chi,double nu,double *v) {
 cb_stream_t*s=cb_stream_create(0,0,1729,0,64);cb_stream_set_stats(s,chi,nu);
 v[0]=s->regime;v[1]=s->eta_star;v[2]=s->sigma;
#ifndef CF_BASELINE
 v[3]=s->cf_radius;v[4]=s->cf_min_curv;v[5]=s->cf_max_curv;
 v[6]=s->cf_derivative;v[7]=s->cf_shift;v[8]=s->cf_sd;v[9]=s->cf_pad;
 v[10]=s->cf_left_f;v[11]=s->cf_right_f;v[12]=s->cf_left_s;v[13]=s->cf_right_s;
 v[14]=s->cf_center_mass;v[15]=s->cf_left_mass;
#endif
 int ok=s->regime==2;cb_stream_destroy(s);return ok;
}
int cf_samples(double chi,double nu,int n,double*x,int path,double *v) {
 cb_stream_t*s=cb_stream_create(0,0,1729,0,64);cb_stream_set_stats(s,chi,nu);
 if(s->regime!=2){cb_stream_destroy(s);return 0;}
 int ok=1;
 if(path==0)ok=cb_stream_draw(s,n,x)==n;
 else if(path==1)for(int i=0;i<n&&ok;++i)ok=cb_stream_draw(s,1,x+i)==1;
#ifndef CF_BASELINE
 else if(path==2){s->cf_left_f+=100.0;ok=draw_cf_batch(s,n,x);}
 else if(path==3){s->cf_pad=1e6;ok=cb_stream_draw(s,n,x)==n;}
#ifdef CB_ARS_DIAGNOSTICS
 v[0]=s->cf_proposals;v[1]=s->cf_evaluations;v[2]=s->cf_squeezes;v[3]=s->cf_tails;
#endif
#endif
 cb_stream_destroy(s);return ok;
}
int cf_benchmark(double chi,double nu,int n,int reps,int path,double *v) {
 cb_stream_t*s=cb_stream_create(0,0,1729,0,64);cb_stream_set_stats(s,chi,nu);
 double *x=malloc((size_t)n*sizeof(double));double start=seconds();int ok=1;
 for(int k=0;k<reps&&ok;++k){
  if(path==2) {cb_stream_set_stats(s,chi,nu);continue;}
  if(path==5 || path==6){
   if(path==6)cb_stream_set_stats(s,chi,nu);
   ok=cb_stream_draw(s,n,x)==n;continue;
  }
  if(path==3 || path==4)cb_stream_set_stats(s,chi,nu);
  if(path==1 || path==4) {s->regime=3;ok=draw_ars_batch(s,n,x);}
  else {
#ifdef CF_BASELINE
   draw_cf_batch(s,n,x);
#else
   ok=draw_cf_batch(s,n,x);
#endif
  }
 }
 v[0]=(seconds()-start)/reps;
 /* Consume the final output after stopping the clock. */
 if(path!=2 && ok){volatile double sink=0;for(int i=0;i<n;++i)sink+=x[i];(void)sink;}
#ifndef CF_BASELINE
#ifdef CB_ARS_DIAGNOSTICS
 v[1]=s->cf_proposals/(double)reps/n;v[2]=s->cf_evaluations/(double)reps/n;
 v[3]=s->cf_squeezes/(double)reps/n;v[4]=s->cf_tails/(double)reps/n;
#endif
#endif
 free(x);cb_stream_destroy(s);return ok;
}
#ifndef CF_BASELINE
int cf_relative(double chi,double nu,int n,const double*x,double*f) {
 cb_stream_t*s=cb_stream_create(0,0,1729,0,64);cb_stream_set_stats(s,chi,nu);
 if(s->regime!=2){cb_stream_destroy(s);return 0;}
 for(int i=0;i<n;++i)f[i]=target_relative(s,x[i]);
 cb_stream_destroy(s);return 1;
}
#endif
