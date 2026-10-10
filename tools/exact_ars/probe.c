/* Private diagnostic harness: compiled by analyze.py, never shipped in the API. */
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include "CORE_FILE"
static double seconds(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
void inspect_hull(double chi,double nu,int insert,double *out){
 cb_stream_t*s=cb_stream_create(0,0,1729,0,64);cb_stream_set_stats(s,chi,nu);s->regime=CB_REGIME_ARS;s->eta_star=cb_mode(s->chi_eff,nu);s->sigma=1/sqrt(nu*cb_bft_d2(s->eta_star));s->buf_target=(insert&2)?1:64;build_hull_exact(s);
#ifndef CB_BENCH_BASELINE
 if(insert&1)for(int k=0;k<15;++k)exact_hull_insert(s,&s->hull,s->eta_star+s->sigma*(-4.0+8.0*(k+.37)/15.0));
#endif
 cb_hull_t*h=&s->hull;out[0]=h->m;out[1]=s->eta_star;out[2]=s->sigma;
#ifdef CB_BENCH_BASELINE
 out[3]=0;
#else
 out[3]=h->padding;
#endif
 for(int i=0;i<h->m;++i){double*v=out+4+8*i;v[0]=h->pts[i];v[1]=h->fvals[i];v[2]=h->slopes[i];v[3]=h->intercepts[i];v[4]=i<h->m-1?h->zpts[i]:0;v[5]=h->logwts[i];
#ifdef CB_BENCH_BASELINE
 v[6]=1;v[7]=i;
#else
 v[6]=h->alias_prob[i];v[7]=h->alias_idx[i];
#endif
}
 cb_stream_destroy(s);
}
void benchmark(double chi,double nu,int n,int reps,int setup_only,double*out){
 cb_stream_t*s=cb_stream_create(0,0,1729,0,64);cb_stream_set_stats(s,chi,nu);s->regime=CB_REGIME_ARS;s->eta_star=cb_mode(s->chi_eff,nu);s->sigma=1/sqrt(nu*cb_bft_d2(s->eta_star));
 double *x=malloc((size_t)n*sizeof(double));double start=seconds();int max_m=0;
 for(int r=0;r<reps;++r){
  s->buf_target=n<=4?1:64;
  build_hull_exact(s);
#ifndef CB_BENCH_BASELINE
  if(setup_only>=2){
   cb_hull_t*h=&s->hull;h->m=2;
   h->pts[0]=hull_drop_point(s,-1,8);h->pts[1]=hull_drop_point(s,1,8);
   for(int i=0;i<2;++i){h->fvals[i]=target_log_f(s->chi_eff,nu,h->pts[i]);h->slopes[i]=target_log_slope(s->chi_eff,nu,h->pts[i]);h->intercepts[i]=target_tangent_intercept(nu,h->pts[i]);}
   s->hull_valid=exact_hull_tables(h);
  }
#endif
  if(setup_only!=1 && setup_only!=3){
#ifdef CB_BENCH_BASELINE
   draw_ars_batch(s,n,x);
#else
   if(!draw_ars_batch(s,n,x)){fprintf(stderr,"draw failure %.17g %.17g\n",chi,nu);free(x);cb_stream_destroy(s);out[0]=-1;return;}
#endif
  }
  if(s->hull.m>max_m)max_m=s->hull.m;
 }
 out[0]=(seconds()-start)/reps;
#ifdef CB_ARS_DIAGNOSTICS
 out[1]=s->ars_proposals/(double)reps/n;
 out[2]=s->ars_insertions/(double)reps;out[3]=s->ars_evaluations/(double)reps;out[4]=max_m;
#else
 out[1]=out[2]=out[3]=0;out[4]=max_m;
#endif
 free(x);cb_stream_destroy(s);
}

#ifndef CB_BENCH_BASELINE
/* Includes copying the retained hull as well as insertion and table refresh. */
void insertion_cost(double chi,double nu,int reps,double *out) {
    cb_stream_t *s=cb_stream_create(0,0,1729,0,64);
    cb_stream_set_stats(s,chi,nu);
    s->eta_star=cb_mode(s->chi_eff,nu);
    s->sigma=1/sqrt(nu*cb_bft_d2(s->eta_star));
    s->buf_target=64;
    build_hull_exact(s);
    cb_hull_t base=s->hull;
    double eta=s->eta_star+.37*s->sigma;
    volatile int inserted=0;
    double start=seconds();
    for (int i=0;i<reps;++i) {
        s->hull=base;
        inserted+=exact_hull_insert(s,&s->hull,eta);
    }
    out[0]=(seconds()-start)/reps;
    out[1]=inserted;
    cb_stream_destroy(s);
}
#endif
