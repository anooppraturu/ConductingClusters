#include "copyright.h"
/*============================================================================*/
/*! \file streaming2d_multi.c
 *  \brief Problem generator for non-linear streaming instability in
 *   non-stratified disks.
 *
 * PURPOSE: Problem generator for non-linear streaming instability in
 *   non-stratified disks. This code works in 2D ONLY. It generalizes the NSH
 *   equilibrium solution to allow multiple-species dust components. Isothermal
 *   eos is assumed, and the value etavk/iso_sound is fixed.
 *
 * Perturbation modes:
 * -  ipert = 0: multi-nsh equilibtium
 * -  ipert = 1: white noise within each grid cell
 * -  ipert = 2: white noise within the entire grid
 * -  ipert = 3: non-nsh velocity
 *
 *  Must be configured using --enable-shearing-box and --with-eos=isothermal.
 *  FARGO is recommended.
 *
 * Reference:
 * - Johansen & Youdin, 2007, ApJ, 662, 627
 * - Bai & Stone, 2009, in preparation */
/*============================================================================*/

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"
#include "particles/particle.h"

#ifndef SHEARING_BOX
#error : The streaming2d problem requires shearing-box to be enabled.
#endif /* SHEARING_BOX */

#ifndef PARTICLES
#error : The streaming2d problem requires particles to be enabled.
#endif /* PARTICLES */

#ifndef ISOTHERMAL
#error : The streaming2d problem requires isothermal equation of state.
#endif /* ISOTHERMAL */

/*------------------------ filewide global variables -------------------------*/
/* NSH equilibrium/perturbation parameters */
Real rho0, mratio, etavk, *epsilon, uxNSH, uyNSH, *wxNSH, *wyNSH;
int ipert;
/* particle number related variables */
int Npar,Npar2;
/* output variables */
long ntrack;   /* number of tracer particles */
long nlis;     /* number of output particles for list output */
int mytype;    /* specific particle type for output */

/*==============================================================================
 * PRIVATE FUNCTION PROTOTYPES:
 * ran2()            - random number generator
 * MultiNSH()        - multiple component NSH equilibrium solver
 * ShearingBoxPot()  - shearing box tidal gravitational potential
 * hst_rho_Vx_dVy()  - total Reynolds stress for history dump
 * property_???()    - particle property selection function
 *============================================================================*/
double ran2(long int *idum);

void MultiNSH(int n, Real *tstop, Real *epsilon, Real etavk,
                     Real *uxNSH, Real *uyNSH, Real *wxNSH, Real *wyNSH);

static Real hst_rho_Vx_dVy(const GridS *pG,const int i,const int j,const int k);

static Real UnstratifiedDisk(const Real x1, const Real x2, const Real x3);

static int property_limit(const GrainS *gr, const GrainAux *grsub);
static int property_trace(const GrainS *gr, const GrainAux *grsub);
static int property_type(const GrainS *gr, const GrainAux *grsub);

extern Real expr_dpar(const GridS *pG, const int i, const int j, const int k);
extern Real expr_V1par(const GridS *pG, const int i, const int j, const int k);
extern Real expr_V2par(const GridS *pG, const int i, const int j, const int k);
extern Real expr_V3par(const GridS *pG, const int i, const int j, const int k);
extern Real expr_V2(const GridS *pG, const int i, const int j, const int k);

/*=========================== PUBLIC FUNCTIONS =================================
 *============================================================================*/
/*----------------------------------------------------------------------------*/
/* problem:   */

void problem(DomainS *pDomain)
{
  GridS *pGrid = pDomain->Grid;
  int i,j,ip,jp,pt;
  long p;
  Real x1,x2,x3,t,x1l,x1u,x2l,x2u,x1p,x2p,x3p;
  Real x1min,x1max,x2min,x2max,Lx,Lz;
  Real tsmin,tsmax,tscrit,pwind,epsum;
  long int iseed = myID_Comm_world; /* Initialize on the first call to ran2 */

  if (pDomain->Nx[1] == 1) {
    ath_error("[streaming2d]: streaming2D only works for 2D problem.\n");
  }
  if (pDomain->Nx[2] > 1) {
    ath_error("[streaming2d]: streaming2D only works for 2D problem.\n");
  }

/* Initialize boxsize */

  x1min = pGrid->MinX[0];
  x1max = pGrid->MaxX[0];
  Lx = x1max - x1min;

  x2min = pGrid->MinX[1];
  x2max = pGrid->MaxX[1];
  Lz = x2max - x2min;

/* Read initial conditions */
  rho0 = 1.0;
  Omega_0 = par_getd("problem","omega");
  qshear = par_getd_def("problem","qshear",1.5);
  ipert = par_geti_def("problem","ipert", 2);
  etavk = par_getd_def("problem","etavk",0.05);/* in unit of iso_sound (N.B.) */

  /* particle number */
  Npar  = (int)(sqrt(par_geti("particle","parnumcell")));
  Npar2 = SQR(Npar);

  pGrid->nparticle = Npar2*pGrid->Nx[0]*pGrid->Nx[1];
  for (i=0; i<npartypes; i++)
    grproperty[i].num = pGrid->nparticle;
  pGrid->nparticle = npartypes*pGrid->nparticle;

  if (pGrid->nparticle+2 > pGrid->arrsize)
    particle_realloc(pGrid, pGrid->nparticle+2);

  /* particle stopping time */
  tsmin = par_getd("problem","tsmin"); /* in code unit */
  tsmax = par_getd("problem","tsmax");
  tscrit= par_getd("problem","tscrit");
  if (par_geti("particle","tsmode") != 3)
    ath_error("[streaming2d]: This test works only for fixed stopping time!\n");

  for (i=0; i<npartypes; i++) {
    tstop0[i] = tsmin*exp(i*log(tsmax/tsmin)/MAX(npartypes-1,1.0));

    /* use fully implicit integrator for well coupled particles */
    if (tstop0[i] < tscrit) grproperty[i].integrator = 3;
  }

  /* assign particle effective mass */
  epsilon= (Real*)calloc_1d_array(npartypes, sizeof(Real));
  wxNSH  = (Real*)calloc_1d_array(npartypes, sizeof(Real));
  wyNSH  = (Real*)calloc_1d_array(npartypes, sizeof(Real));

#ifdef FEEDBACK
  mratio = par_getd_def("problem","mratio",0.0); /* total mass fraction */
  pwind = par_getd_def("problem","pwind",0.0);
  if (mratio < 0.0)
    ath_error("[streaming2d]: mratio must be positive!\n");

  epsum = 0.0;
  for (i=0; i<npartypes; i++)
  {
    epsilon[i] = pow(tstop0[i],pwind);	epsum += epsilon[i];
  }

  for (i=0; i<npartypes; i++)
  {
    epsilon[i] = mratio*epsilon[i]/epsum;
    grproperty[i].m = rho0*epsilon[i]/Npar2;
  }
#else
  mratio = 0.0;
  for (i=0; i<npartypes; i++)
    epsilon[i] = 0.0;
#endif

  etavk = etavk * Iso_csound; /* switch to code unit */

  /* calculate NSH equilibrium velocity */
  MultiNSH(npartypes, tstop0, epsilon, etavk, &uxNSH, &uyNSH, wxNSH, wyNSH);

/* Now set initial conditions for the gas */
  for (j=pGrid->js; j<=pGrid->je; j++) {
  for (i=pGrid->is; i<=pGrid->ie; i++) {
    cc_pos(pGrid,i,j,pGrid->ks,&x1,&x2,&x3);

    pGrid->U[pGrid->ks][j][i].d = rho0;

    if (ipert != 3) {
      pGrid->U[pGrid->ks][j][i].M1 = rho0 * uxNSH;
      pGrid->U[pGrid->ks][j][i].M3 = rho0 * uyNSH;
    } else {
      pGrid->U[pGrid->ks][j][i].M1 = 0.0;
      pGrid->U[pGrid->ks][j][i].M3 = 0.0;
    }

    pGrid->U[pGrid->ks][j][i].M2 = 0.0;
#ifndef FARGO
    pGrid->U[pGrid->ks][j][i].M3 -= qshear*rho0*Omega_0*x1;
#endif

  }}

/* Now set initial conditions for the particles */
  p = 0;
  x3p = 0.5*(pGrid->MinX[2]+pGrid->MaxX[2]);

  for (j=pGrid->js; j<=pGrid->je; j++)
  {
    x2l = pGrid->MinX[1] + (j-pGrid->js)*pGrid->dx2;
    x2u = pGrid->MinX[1] + (j-pGrid->js+1.0)*pGrid->dx2;

    for (i=pGrid->is; i<=pGrid->ie; i++)
    {
      x1l = pGrid->MinX[0] + (i-pGrid->is)*pGrid->dx1;
      x1u = pGrid->MinX[0] + (i-pGrid->is+1.0)*pGrid->dx1;

        for (pt=0; pt<npartypes; pt++)
        {
          for (ip=0;ip<Npar;ip++)
          {

            if (ipert == 0)
              x1p = x1l+pGrid->dx1/Npar*(ip+0.5);

            for (jp=0;jp<Npar;jp++)
            {
              if (ipert == 0)
                x2p = x2l+pGrid->dx2/Npar*(jp+0.5);
              else if (ipert == 1)
              { /* ramdom particle position in the grid */
                x1p = x1l + pGrid->dx1*ran2(&iseed);
                x2p = x2l + pGrid->dx2*ran2(&iseed);
              }
              else
              {
                x1p = x1min + Lx*ran2(&iseed);
                x2p = x2min + Lz*ran2(&iseed);
              }

              pGrid->particle[p].property = pt;
              pGrid->particle[p].x1 = x1p;
              pGrid->particle[p].x2 = x2p;
              pGrid->particle[p].x3 = x3p;

              if (ipert != 3) {
                pGrid->particle[p].v1 = wxNSH[pt];
                pGrid->particle[p].v3 = wyNSH[pt];
              } else {
                pGrid->particle[p].v1 = 0.0;
                pGrid->particle[p].v3 = etavk;
              }

              pGrid->particle[p].v2 = 0.0;
#ifndef FARGO
              pGrid->particle[p].v3 -= qshear*Omega_0*x1p;
#endif
              pGrid->particle[p].pos = 1; /* grid particle */
              pGrid->particle[p].my_id = p;
#ifdef MPI_PARALLEL
              pGrid->particle[p].init_id = myID_Comm_world;
#endif
              p += 1;
            }
          }
        }
    }
  }

/* enroll gravitational potential function, shearing sheet BC functions */
  ShearingBoxPot = UnstratifiedDisk;

  ShBoxCoord = xz;

  dump_history_enroll(hst_rho_Vx_dVy, "<rho Vx dVy>");

  /* set the # of the particles in list output
   * (by default, output 1 particle per cell)
   */
  nlis = par_geti_def("problem","nlis",pGrid->Nx[0]*pGrid->Nx[1]);

  /* set the number of particles to keep track of */
  ntrack = par_geti_def("problem","ntrack",2000);

  return;
}

/*==============================================================================
 * PROBLEM USER FUNCTIONS:
 * problem_write_restart() - writes problem-specific user data to restart files
 * problem_read_restart()  - reads problem-specific user data from restart files
 * get_usr_expr()          - sets pointer to expression for special output data
 * get_usr_out_fun()       - returns a user defined output function pointer
 * Userwork_in_loop        - problem specific work IN     main loop
 * Userwork_after_loop     - problem specific work AFTER  main loop
 *----------------------------------------------------------------------------*/

void problem_write_restart(MeshS *pM, FILE *fp)
{
  fwrite(&rho0, sizeof(Real),1,fp);
  fwrite(&mratio, sizeof(Real),1,fp); fwrite(epsilon, sizeof(Real),npartypes,fp);
  fwrite(&etavk, sizeof(Real),1,fp);
  fwrite(&uxNSH, sizeof(Real),1,fp);
  fwrite(&uyNSH, sizeof(Real),1,fp);
  fwrite(wxNSH, sizeof(Real),npartypes,fp);
  fwrite(wyNSH, sizeof(Real),npartypes,fp);

  return;
}

void problem_read_restart(MeshS *pM, FILE *fp)
{
  DomainS *pD = (DomainS*)&(pM->Domain[0][0]);
  GridS *pG = pD->Grid;
  Real singleintvl,tracetime,traceintvl;

  ShearingBoxPot = UnstratifiedDisk;
  ShBoxCoord = xz;

  Omega_0 = par_getd("problem","omega");
  qshear = par_getd_def("problem","qshear",1.5);
  ipert = par_geti_def("problem","ipert", 1);

  Npar  = (int)(sqrt(par_geti("particle","parnumcell")));
  Npar2 = SQR(Npar);
  nlis = par_geti_def("problem","nlis",pG->Nx[0]*pG->Nx[1]);
  ntrack = par_geti_def("problem","ntrack",2000);

  /* assign particle effective mass */
  epsilon= (Real*)calloc_1d_array(npartypes, sizeof(Real));
  wxNSH  = (Real*)calloc_1d_array(npartypes, sizeof(Real));
  wyNSH  = (Real*)calloc_1d_array(npartypes, sizeof(Real));

  fread(&rho0, sizeof(Real),1,fp);
  fread(&mratio, sizeof(Real),1,fp);
  fread(epsilon,sizeof(Real),npartypes,fp);
  fread(&etavk, sizeof(Real),1,fp);
  fread(&uxNSH, sizeof(Real),1,fp);
  fread(&uyNSH, sizeof(Real),1,fp);
  fread(wxNSH, sizeof(Real),npartypes,fp);
  fread(wyNSH, sizeof(Real),npartypes,fp);

  dump_history_enroll(hst_rho_Vx_dVy, "<rho Vx dVy>");

  return;
}

/*! \fn static Real expr_rhopardif(const GridS *pG,
 *                           const int i, const int j, const int k)
 *  \brief Particle density difference */
static Real expr_rhopardif(const GridS *pG,
                           const int i, const int j, const int k)
{
  Real x1,x2,x3;
  cc_pos(pG,i,j,k,&x1,&x2,&x3);
  return pG->Coup[k][j][i].grid_d - rho0*mratio;
}

/*----------------------------------------------------------------------------*/
/*! \fn static Real hst_rho_Vx_dVy(const GridS *pG,
 *                         const int i, const int j, const int k)
 *  \brief Reynolds stress, added as history variable.
 */

static Real hst_rho_Vx_dVy(const GridS *pG,
                           const int i, const int j, const int k)
{
  Real x1,x2,x3;
  cc_pos(pG,i,j,k,&x1,&x2,&x3);
#ifdef FARGO
  return pG->U[k][j][i].M1*pG->U[k][j][i].M3/pG->U[k][j][i].d;
#else
  return pG->U[k][j][i].M1*(pG->U[k][j][i].M3/pG->U[k][j][i].d
                            + qshear*Omega_0*x1);
#endif
}

ConsFun_t get_usr_expr(const char *expr)
{
  if(strcmp(expr,"difdpar")==0) return expr_rhopardif;
  return NULL;
}

VOutFun_t get_usr_out_fun(const char *name){
  return NULL;
}

#ifdef PARTICLES
PropFun_t get_usr_par_prop(const char *name)
{
  if (strcmp(name,"limit")==0)    return property_limit;
  if (strcmp(name,"trace")==0)    return property_trace;
  if (strcmp(name,"type")==0)    return property_type;
  return NULL;
}

/*! \fn void gasvshift(const Real x1, const Real x2, const Real x3,
 *                                  Real *u1, Real *u2, Real *u3)
 *  \brief Gas velocity shift */
void gasvshift(const Real x1, const Real x2, const Real x3,
                                    Real *u1, Real *u2, Real *u3)
{
  return;
}

void Userforce_particle(Real3Vect *ft, const Real x1, const Real x2, const Real x3,
                                    const Real v1, const Real v2, const Real v3)
{
  ft->x1 -= 2.0*etavk*Omega_0;
  return;
}
#endif

void Userwork_in_loop(MeshS *pM)
{
  return;
}

/*---------------------------------------------------------------------------
 * Userwork_after_loop
 */

void Userwork_after_loop(MeshS *pM)
{
  free(epsilon);
  free(wxNSH);    free(wyNSH);
  return;
}

/*=========================== PRIVATE FUNCTIONS ==============================*/
/*--------------------------------------------------------------------------- */
/*! \fn static Real UnstratifiedDisk(const Real x1, const Real x2,const Real x3)
 *  \brief shearing box tidal gravitational potential*/
static Real UnstratifiedDisk(const Real x1, const Real x2, const Real x3)
{
  Real phi=0.0;
#ifndef FARGO
  phi -= qshear*SQR(Omega_0*x1);
#endif
  return phi;
}

/*! \fn static int property_limit(const GrainS *gr, const GrainAux *grsub)
 *  \brief user defined particle selection function (1: true; 0: false) */
static int property_limit(const GrainS *gr, const GrainAux *grsub)
{
  if ((gr->pos == 1) && (gr->my_id<nlis))
    return 1;
  else
    return 0;
}

/*! \fn static int property_trace(const GrainS *gr, const GrainAux *grsub)
 *  \brief user defined particle selection function (1: true; 0: false) */
static int property_trace(const GrainS *gr, const GrainAux *grsub)
{
  if ((gr->pos == 1) && (gr->my_id<ntrack))
    return 1;
  else
    return 0;
}

/*! \fn static int property_type(const GrainS *gr, const GrainAux *grsub)
 *  \brief user defined particle selection function (1: true; 0: false) */
static int property_type(const GrainS *gr, const GrainAux *grsub)
{
  if ((gr->pos == 1) && (gr->property == mytype))
    return 1;
  else
    return 0;
}

/*--------------------------------------------------------------------------- */
/*! \fn void MultiNSH(int n, Real *tstop, Real *mratio, Real etavk,
 *                   Real *uxNSH, Real *uyNSH, Real *wxNSH, Real *wyNSH)
 *  \brief Multi-species NSH equilibrium
 *
 * Input: # of particle types (n), dust stopping time and mass ratio array, and
 *        drift speed etavk.
 * Output: gas NSH equlibrium velocity (u), and dust NSH equilibrium velocity
 *         array (w).
 */
void MultiNSH(int n, Real *tstop, Real *mratio, Real etavk,
                     Real *uxNSH, Real *uyNSH, Real *wxNSH, Real *wyNSH)
{
  int i,j;
  Real *Lambda1,**Lam1GamP1, **A, **B, **Tmp;

  Lambda1 = (Real*)calloc_1d_array(n, sizeof(Real));     /* Lambda^{-1} */
  Lam1GamP1=(Real**)calloc_2d_array(n, n, sizeof(Real));/* Lambda1*(1+Gamma) */
  A       = (Real**)calloc_2d_array(n, n, sizeof(Real));
  B       = (Real**)calloc_2d_array(n, n, sizeof(Real));
  Tmp     = (Real**)calloc_2d_array(n, n, sizeof(Real));

  /* definitions */
  for (i=0; i<n; i++){
    for (j=0; j<n; j++)
      Lam1GamP1[i][j] = mratio[j];
    Lam1GamP1[i][i] += 1.0;
    Lambda1[i] = 1.0/(tstop[i]+1.0e-16);
    for (j=0; j<n; j++)
      Lam1GamP1[i][j] *= Lambda1[i];
  }

  /* Calculate A and B */
  MatrixMult(Lam1GamP1, Lam1GamP1, n,n,n, Tmp);
  for (i=0; i<n; i++) Tmp[i][i] += 1.0;
  InverseMatrix(Tmp, n, B);
  for (i=0; i<n; i++)
  for (j=0; j<n; j++)
    B[i][j] *= Lambda1[j];
  MatrixMult(Lam1GamP1, B, n,n,n, A);

  /* Obtain NSH velocities */
  *uxNSH = 0.0;  *uyNSH = 0.0;
  for (i=0; i<n; i++){
    wxNSH[i] = 0.0;
    wyNSH[i] = 0.0;
    for (j=0; j<n; j++){
      wxNSH[i] -= B[i][j];
      wyNSH[i] -= A[i][j];
    }
    wxNSH[i] *= 2.0*etavk;
    wyNSH[i] *= etavk;
    *uxNSH -= mratio[i]*wxNSH[i];
    *uyNSH -= mratio[i]*wyNSH[i];
    wyNSH[i] += etavk;
  }

  free(Lambda1);
  free_2d_array(A);         free_2d_array(B);
  free_2d_array(Lam1GamP1); free_2d_array(Tmp);

  return;
}

/*------------------------------------------------------------------------------
 */

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define RNMX (1.0-DBL_EPSILON)

/*! \fn double ran2(long int *idum)
 *  \brief Extracted from the Numerical Recipes in C (version 2) code.  Modified
 *   to use doubles instead of floats. -- T. A. Gardiner -- Aug. 12, 2003
 *
 * Long period (> 2 x 10^{18}) random number generator of L'Ecuyer
 * with Bays-Durham shuffle and added safeguards.  Returns a uniform
 * random deviate between 0.0 and 1.0 (exclusive of the endpoint
 * values).  Call with idum = a negative integer to initialize;
 * thereafter, do not alter idum between successive deviates in a
 * sequence.  RNMX should appriximate the largest floating point value
 * that is less than 1.
 */

double ran2(long int *idum)
{
  int j;
  long int k;
  static long int idum2=123456789;
  static long int iy=0;
  static long int iv[NTAB];
  double temp;

  if (*idum <= 0) { /* Initialize */
    if (-(*idum) < 1) *idum=1; /* Be sure to prevent idum = 0 */
    else *idum = -(*idum);
    idum2=(*idum);
    for (j=NTAB+7;j>=0;j--) { /* Load the shuffle table (after 8 warm-ups) */
      k=(*idum)/IQ1;
      *idum=IA1*(*idum-k*IQ1)-k*IR1;
      if (*idum < 0) *idum += IM1;
      if (j < NTAB) iv[j] = *idum;
    }
    iy=iv[0];
  }
  k=(*idum)/IQ1;                 /* Start here when not initializing */
  *idum=IA1*(*idum-k*IQ1)-k*IR1; /* Compute idum=(IA1*idum) % IM1 without */
  if (*idum < 0) *idum += IM1;   /* overflows by Schrage's method */
  k=idum2/IQ2;
  idum2=IA2*(idum2-k*IQ2)-k*IR2; /* Compute idum2=(IA2*idum) % IM2 likewise */
  if (idum2 < 0) idum2 += IM2;
  j=(int)(iy/NDIV);              /* Will be in the range 0...NTAB-1 */
  iy=iv[j]-idum2;                /* Here idum is shuffled, idum and idum2 */
  iv[j] = *idum;                 /* are combined to generate output */
  if (iy < 1) iy += IMM1;
  if ((temp=AM*iy) > RNMX) return RNMX; /* No endpoint values */
  else return temp;
}

#undef IM1
#undef IM2
#undef AM
#undef IMM1
#undef IA1
#undef IA2
#undef IQ1
#undef IQ2
#undef IR1
#undef IR2
#undef NTAB
#undef NDIV
#undef RNMX
