#include "copyright.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"

#include "prob/math_functions.h"

#ifdef MPI_PARALLEL
#include "mpi.h"
#ifdef DOUBLE_PREC
#define MPI_RL MPI_DOUBLE
#else /* DOUBLE_PREC */
#define MPI_RL MPI_FLOAT
#endif /* DOUBLE_PREC */
#endif /* MPI_PARALLEL */

/* ========================================================================== */
/* Prototypes and Definitions */

/* Global Variables for use in Potential and Boundary Conditions */
/* -------------------------------------------------------------------------- */
static Real t_ofst; /* pM->time = 0 corresponds to a finite cosmic time */
static Real c_nfw, f, g, b, h, z, m, rvir, rho_out;
static Real rout, rsoft; /* soften gravity at large and small radii */
static Real xi_init, Tigm;
static Real mu=0.62, mue=1.17;
static Real m15;
static int cooling=0;
static Real mytanh(const Real x);

/* Cosmology */
/* -------------------------------------------------------------------------- */
static const Real Om=0.27, OL=0.73, fb=0.17;
static Real zz(Real t);
static Real hh(Real z1);
static Real mm(Real z1);

static Real phi_nfw(const Real r);
static void inner_bc(DomainS *pDomain);


/* Read in the initial condition from a file... */
/* -------------------------------------------------------------------------- */
static void import_atmosphere(char *atm_file);
static int n_entries;
static Real *x_init, *v_init, *d_init;


/* Physics */
/* -------------------------------------------------------------------------- */
static void set_vars(Real time);       /* use in problem() and read_restart() */

static Real grav_pot(const Real x1, const Real x2, const Real x3, const Real time);

#ifdef THERMAL_CONDUCTION
static Real kappa_fun(const Real d, const Real T,
                      const Real x1, const Real x2, const Real x3);

static Real f_sp;             /* normalization for the conductivity */
#endif  /* THERMAL_CONDUCTION */


#ifdef VISCOSITY
/* viscosity won't work without conduction... */
static Real nu_fun(const Real d, const Real T,
                   const Real x1, const Real x2, const Real x3);
#endif  /* VISCOSITY */

static Real kT_keV(const Real P, const Real d);
static Real L_23(const Real T);
static Real cool(const Real d, const Real P, const Real dt);
static void integrate_cool(DomainS *pDomain, const Real dt_hydro);
static void cool_step(GridS *pGrid, const Real dt);


/* Making radial profiles*/
/* -------------------------------------------------------------------------- */
static OutputS profile_dump;
void dump_profile(DomainS *pD, OutputS *pOut);


static Real **profile_data;
static int n_bins;
/*num, radius, P, rho, T, K, Mach, Vr, Metals, Convective Heat Flux, B^2, beta, Br, Alfven_Mach*/
#ifndef MHD
const int n_profiles = 10;
#else
const int n_profiles = 14;
#endif /*MHD*/

#ifdef MPI_PARALLEL
static Real **profile_data_global;
#endif  /* MPI_PARALLEL */

static void calc_profiles(DomainS *pDomain, Real **profile_data);

/*Hisory Dump Stuff*/
/* ------------------------------------------------------------------------ */
static Real Metal_r2(const GridS *pG, const int i, const int j, const int k);


/* FFT prototypes */
/* ------------------------------------------------------------------------ */

/* FFT indexing Nfast=k, Nmid=j, Nslow=i (opposite to Athena)
 * For OFST, i,j,k,nx2,nx3 reference the local grid */
#define OFST(i, j, k) ((k) + nx3*((j) + nx2*(i)))
/* KWVM: magnitude of wavenumber k in units of dkx */
#define KWVM(i, j, k) (sqrt(SQR(KCOMP(i,gis-nghost,gnx1))+ \
                            SQR(KCOMP(j,gjs-nghost,gnx2))+ \
                            SQR(KCOMP(k,gks-nghost,gnx3))))

/* FFTW - Variables, Plan, etc. */
static struct ath_3d_fft_plan *plan;

static ath_fft_data *fA1=NULL, *fA2=NULL, *fA3=NULL;
static Real ***A1=NULL, ***A2=NULL, ***A3=NULL;

/* Cutoff wavenumbers, G&O spect peak, power law spect exponent, 2 pi/L */
static Real klow,khigh,expo,dkx;
/* Number of cells in local grid, number of cells in global grid */
static int nx1,nx2,nx3,gnx1,gnx2,gnx3;
/* Starting and ending indices for global grid */
static int gis,gie,gjs,gje,gks,gke;
/* Seed for random number generator */
long int rseed;

/* Functions appear in this file in the same order that they appear in the
 * prototypes below */

/* Function prototypes for generating velocity perturbations */
static void pspect(ath_fft_data *ampl);
static inline void transform();
static inline void generate();
static void perturb(DomainS *pDomain);

/* Function prototypes for initializing and interfacing with Athena */
static void initialize(GridS *pGrid, DomainS *pD);

/* Function prototypes for Numerical Recipes functions */
static double ran2(long int *idum);



/* end prototypes and definitions */
/* ========================================================================== */


/* ========================================================================== */
/* Before we get to the problem file, FFT functions for B */

/*  Power spectrum returned in ampl
 *  - klow   = multiple of 2 pi/L for cut-off at low  wavenumbers
 *  - khigh  = multiple of 2 pi/L for cut-off at high wavenumbers
 *  - expo   = exponent of power law
 *  - ispect = integer flag which specifies spectrum
 *
 *  Note that the fourier amplitudes are stored in an array with no
 *  ghost zones
 */
static void pspect(ath_fft_data *ampl)
{
  int i,j,k;
  double q1,q2,q3;

  /* set random amplitudes with gaussian deviation */
  for (k=0; k<nx3; k++) {
    for (j=0; j<nx2; j++) {
      for (i=0; i<nx1; i++) {
        q1 = ran2(&rseed);
        q2 = ran2(&rseed);
        q3 = sqrt(-2.0*log(q1+1.0e-20))*cos(2.0*PI*q2);
        q1 = ran2(&rseed);
        ampl[OFST(i,j,k)][0] = q3*cos(2.0*PI*q1);
        ampl[OFST(i,j,k)][1] = q3*sin(2.0*PI*q1);
      }
    }
  }

  /* set power spectrum */
  for (k=0; k<nx3; k++) {
    for (j=0; j<nx2; j++) {
      for (i=0; i<nx1; i++) {
        /* compute k/dkx */
        q3 = KWVM(i,j,k);
        if ((q3 > klow) && (q3 < khigh)) { /* decreasing power law */
          q3 *= dkx; /* multiply by 2 pi/L */

          ampl[OFST(i,j,k)][0] /= pow(q3,(expo+2.0));
          ampl[OFST(i,j,k)][1] /= pow(q3,(expo+2.0));
        } else {                /* introduce cut-offs at klow and khigh */
          ampl[OFST(i,j,k)][0] = 0.0;
          ampl[OFST(i,j,k)][1] = 0.0;
        }

      }
    }
  }
  ampl[0][0] = 0.0;
  ampl[0][1] = 0.0;

  return;
}

static inline void transform()
{
  /* Transform vector potential from k space to physical space */
  ath_3d_fft(plan, fA1);
  ath_3d_fft(plan, fA2);
  ath_3d_fft(plan, fA3);

  /*I think the below is from Jimmy's original turb.c, ignore I guess...*/
  /* Should technically renormalize (divide by gnx1*gnx2*gnx3) here, but
   * since we're going to renormalize to get the desired energy injection
   * rate anyway, there's no point */

  return;
}

static inline void generate()
{
  /* Generate new perturbations following appropriate power spectrum */
  pspect(fA1);
  pspect(fA2);
  pspect(fA3);


  /* Transform perturbations to real space, but don't normalize until
   * just before we apply them in perturb() */
  transform();

  return;
}

static void perturb(DomainS *pDomain)
{
  GridS *pGrid = pDomain->Grid;
  int i, is=pGrid->is, ie = pGrid->ie;
  int j, js=pGrid->js, je = pGrid->je;
  int k, ks=pGrid->ks, ke = pGrid->ke, ku;
  int ind, mpierr;
  Real Bstrength;
  Real rms[2], grms[2];

  Bstrength = par_getd("problem", "Bstrength");

  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        ind = OFST(i-is,j-js,k-ks);
        A1[k][j][i] = fA1[ind][0];
        A2[k][j][i] = fA2[ind][0];
        A3[k][j][i] = fA3[ind][0];
      }
    }
  }

  rms[0] = rms[1] = 0.0;
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        rms[0] += SQR(A1[k][j][i]) + SQR(A2[k][j][i]) + SQR(A3[k][j][i]);
        rms[1] += 1;
      }
    }
  }

#ifdef MPI_PARALLEL
  mpierr = MPI_Allreduce(rms, grms, 2, MPI_RL, MPI_SUM, MPI_COMM_WORLD);
  if (mpierr) ath_error("[normalize]: MPI_Allreduce error = %d\n", mpierr);
  rms[0] = grms[0];
  rms[1] = grms[1];
#endif /* MPI_PARALLEL */

  rms[0] = sqrt(rms[0]/rms[1]);
  //ath_pout(0, "rms = %f\n", rms);

  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        A1[k][j][i] /= pow(10.0,Bstrength)*rms[0];
        A2[k][j][i] /= pow(10.0,Bstrength)*rms[0];
        A3[k][j][i] /= pow(10.0,Bstrength)*rms[0];
      }
    }
  }

#ifdef MPI_PARALLEL
  /* need to populate ghost zones... */
  /* ... copy to A B*c as a hack so that we can use the built-in
     bvals_mhd() to handle interprocessor communication.  */

  /* copy A* to B*c */
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        pGrid->U[k][j][i].B1c = A1[k][j][i];
        pGrid->U[k][j][i].B2c = A2[k][j][i];
        pGrid->U[k][j][i].B3c = A3[k][j][i];
      }
    }
  }

  /* bvals_mhd() for communication */
  bvals_mhd(pDomain);

  /* copy B*c to the ghost-zones of A* */

  /* x1 direction */
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost; i++) {
              A1[k][j][is-i] = pGrid->U[k][j][is-i].B1c;
              A1[k][j][ie+i] = pGrid->U[k][j][ie+i].B1c;

              A2[k][j][is-i] = pGrid->U[k][j][is-i].B2c;
              A2[k][j][ie+i] = pGrid->U[k][j][ie+i].B2c;

              A3[k][j][is-i] = pGrid->U[k][j][is-i].B3c;
              A3[k][j][ie+i] = pGrid->U[k][j][ie+i].B3c;
      }
    }
  }

  /* x2 direction */
  if (pGrid->Nx[1] > 1){
    for (k=ks; k<=ke; k++) {
      for (j=1; j<=nghost; j++) {
         for (i=is-nghost; i<=ie+nghost; i++) {
            A1[k][js-j][i] = pGrid->U[k][js-j][i].B1c;
            A1[k][je+j][i] = pGrid->U[k][je+j][i].B1c;

            A2[k][js-j][i] = pGrid->U[k][js-j][i].B2c;
            A2[k][je+j][i] = pGrid->U[k][je+j][i].B2c;

            A3[k][js-j][i] = pGrid->U[k][js-j][i].B3c;
            A3[k][je+j][i] = pGrid->U[k][je+j][i].B3c;
         }
      }
    }
  }

  /* x3 direction */
  if (pGrid->Nx[2] > 1){
    for (k=1; k<=nghost; k++) {
      for (j=js-nghost; j<=je+nghost; j++) {
         for (i=is-nghost; i<=ie+nghost; i++) {
           A1[ks-k][j][i] = pGrid->U[ks-k][j][i].B1c;
           A1[ke+k][j][i] = pGrid->U[ke+k][j][i].B1c;

           A2[ks-k][j][i] = pGrid->U[ks-k][j][i].B2c;
           A2[ke+k][j][i] = pGrid->U[ke+k][j][i].B2c;

           A3[ks-k][j][i] = pGrid->U[ks-k][j][i].B3c;
           A3[ke+k][j][i] = pGrid->U[ke+k][j][i].B3c;
         }
      }
    }
  }

  /* phew! */
#endif /*MPI_PARALLEL*/

  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie+1; i++) {
         pGrid->B1i[k][j][i] = (A3[k][j+1][i] - A3[k][j][i])/pGrid->dx2 -
                          (A2[k+1][j][i] - A2[k][j][i])/pGrid->dx3;
      }
    }
  }
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je+1; j++) {
      for (i=is; i<=ie; i++) {
         pGrid->B2i[k][j][i] = (A1[k+1][j][i] - A1[k][j][i])/pGrid->dx3 -
                          (A3[k][j][i+1] - A3[k][j][i])/pGrid->dx1;
      }
    }
  }
  if (ke > ks) {
    ku = ke+1;
  } else {
    ku = ke;
  }
  for (k=ks; k<=ku; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
         pGrid->B3i[k][j][i] = (A2[k][j][i+1] - A2[k][j][i])/pGrid->dx1 -
                          (A1[k][j+1][i] - A1[k][j][i])/pGrid->dx2;
      }
    }
  }
  return;
}

static void initialize(GridS *pGrid, DomainS *pD)
{
  int i, is=pGrid->is, ie = pGrid->ie;
  int j, js=pGrid->js, je = pGrid->je;
  int k, ks=pGrid->ks, ke = pGrid->ke;
  int nbuf, mpierr, nx1gh, nx2gh, nx3gh;

/* -----------------------------------------------------------
 * Variables within this block are stored globally, and used
 * within preprocessor macros.  Don't create variables with
 * these names within your function if you are going to use
 * OFST(), KCOMP(), or KWVM() within the function! */

  /* Get local grid size */
  nx1 = (ie-is+1);
  nx2 = (je-js+1);
  nx3 = (ke-ks+1);

  /* Get global grid size */
  gnx1 = pD->Nx[0];
  gnx2 = pD->Nx[1];
  gnx3 = pD->Nx[2];

  /* Get extents of local FFT grid in global coordinates */
  gis=is+pGrid->Disp[0];  gie=ie+pGrid->Disp[0];
  gjs=js+pGrid->Disp[1];  gje=je+pGrid->Disp[1];
  gks=ks+pGrid->Disp[2];  gke=ke+pGrid->Disp[2];
/* ----------------------------------------------------------- */

  /* Get size of arrays with ghost cells */
  nx1gh = nx1 + 2*nghost;
  nx2gh = nx2 + 2*nghost;
  nx3gh = nx3 + 2*nghost;

  expo = par_getd("problem","expo");

  /* Cutoff wavenumbers of spectrum */
  klow = par_getd("problem","klow"); /* in integer units */
  khigh = par_getd("problem","khigh"); /* in integer units */
  dkx = 2.0*PI/(pGrid->dx1*gnx1); /* convert k from integer */

  /* Allocate memory for components of vector potential */
  if ((A1=(Real***)calloc_3d_array(nx3gh,nx2gh,nx1gh,sizeof(Real)))==NULL) {
    ath_error("[problem]: Error allocating memory for vel pert\n");
  }
  if ((A2=(Real***)calloc_3d_array(nx3gh,nx2gh,nx1gh,sizeof(Real)))==NULL) {
    ath_error("[problem]: Error allocating memory for vel pert\n");
  }
  if ((A3=(Real***)calloc_3d_array(nx3gh,nx2gh,nx1gh,sizeof(Real)))==NULL) {
    ath_error("[problem]: Error allocating memory for vel pert\n");
  }


  /* Initialize the FFT plan */
  plan = ath_3d_fft_quick_plan(pD, NULL, ATH_FFT_BACKWARD);

  fA1 = ath_3d_fft_malloc(plan);
  fA2 = ath_3d_fft_malloc(plan);
  fA3 = ath_3d_fft_malloc(plan);


  return;
}

/* Create the initial condition and start the simulation! */
void problem(DomainS *pDomain)
{
  GridS *pGrid = pDomain->Grid;
  int i=0,j=0,k=0;
  int is,ie,js,je,ks,ke,iprob;
  int il,iu,jl,ju,kl,ku;
  Real x1,x2,x3, r;
  Real x1m;
  Real rho, P, v, KE, phi;
  Real rhoi, rhoshock, Pshock, phishock, rshock, fact;

  int Nx = pDomain->Nx[0];
  int Ny = pDomain->Nx[1];
  int Nz = pDomain->Nx[2];

  int prof_index,bin_index;

  is = pGrid->is; ie = pGrid->ie;
  js = pGrid->js; je = pGrid->je;
  ks = pGrid->ks; ke = pGrid->ke;

#if (NSCALARS > 0)
  int n;
  Real halfwidth;
#endif

  /* Ensure a different initial random seed for each process in an MPI calc. */
  rseed = -11;
#ifdef MPI_PARALLEL
  rseed -= myID_Comm_world;
#endif
  initialize(pGrid, pDomain);

  set_vars(pGrid->time);

  /* find the shock and calculate the pre- and post-shock
     conditions. */
  z = zz(pGrid->time + t_ofst);
  m = mm(z);
  h = hh(z);
  rvir = pow(m, 1.0/3.0) * pow(10.0*h, -2.0/3.0);
  if (2.0 * rvir >= rout){
    ath_error("[problem]: virial radius exceeds domain size.\n");
  }
  rshock = xi_init * rvir;
  phishock = fabs(phi_nfw(rshock));

  /* -- pre-shock, and... */
  v        = interpolate(x_init, v_init, rshock/rvir, n_entries);
  rhoi     = interpolate(x_init, d_init, rshock/rvir, n_entries) * fb;

  /* -- ...post-shock */
  rhoshock = 4.0/(1.0 + 5.0*Tigm/SQR(v)) * rhoi;
  Pshock   = ((4.0/3.0)*SQR(v) - Tigm/4.0) * rhoi;

  /*Stuff to dump*/
  profile_dump.n      = 100;
  profile_dump.dt     = par_getd("problem", "dt");
  profile_dump.t      = pGrid->time;
  profile_dump.num    = 0;
  profile_dump.out    = "prim";
  profile_dump.nlevel = -1;       /* dump all levels */
  n_bins = MAX(Nx,Ny);
  n_bins = MAX(n_bins,Nz) + 8.0;

  /*Allocate and initialize array to hold profile data*/
  profile_data = (Real**) calloc_2d_array(n_profiles, n_bins, sizeof(Real));
  for(prof_index = 0; prof_index<n_profiles; prof_index++){
     for(bin_index = 0; bin_index<n_bins; bin_index++){
        profile_data[prof_index][bin_index] = 0.0;
     }
  }
#ifdef MPI_PARALLEL
  profile_data_global = (Real**) calloc_2d_array(n_profiles, n_bins, sizeof(Real));

  for(prof_index = 0; prof_index<n_profiles; prof_index++){
    for(bin_index = 0; bin_index<n_bins; bin_index++){
      profile_data_global[prof_index][bin_index] = 0.0;
    }
  }
#endif /*MPI_PARALLEL*/

  /* initialize gas variables on the grid */
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        cc_pos(pGrid,i,j,k,&x1,&x2,&x3);
        r = sqrt(x1*x1 + x2*x2 + x3*x3);

        if (r > rshock){
          /* infall model calculated externally */
          v   = interpolate(x_init, v_init, r/rvir, n_entries);
          rho = interpolate(x_init, d_init, r/rvir, n_entries) * fb;
          P   = Tigm * rho;
        }else{
          /* isentropic core */
          phi  = fabs(phi_nfw(r));
          fact = (2.0*rhoshock)/(5.0*Pshock) * (phi - phishock);

          rho = rhoshock * pow(1.0+fact, 1.5);
          P   = Pshock * pow(rho/rhoshock, 5.0/3.0);
          v   = 0.0;
        }

        pGrid->U[k][j][i].d  = rho;
        pGrid->U[k][j][i].M1 = -1.0 * rho * v * x1/r;
        pGrid->U[k][j][i].M2 = -1.0 * rho * v * x2/r;
        pGrid->U[k][j][i].M3 = -1.0 * rho * v * x3/r;

#if (NSCALARS > 0)
        x1m = pDomain->RootMaxX[0];
        halfwidth = x1m/(2.0*(float)NSCALARS);
	pGrid->U[k][j][i].s[0] = 0.0;

        for(n=1; n<NSCALARS; n++){
           pGrid->U[k][j][i].s[n] = 0.5*(tanh(50.0*(r+halfwidth - halfwidth*(2.0*n+1.0)))+tanh(-50.0*(r-halfwidth-halfwidth*(2.0*n+1))));
        }
#endif
#ifndef ISOTHERMAL
        KE = 0.5 * rho * SQR(v);
        pGrid->U[k][j][i].E = KE + P/Gamma_1;
#endif  /* ISOTHERMAL */
      }
    }
  }

   /*Add metal weighted r^2 to history dump*/
#if (NSCALARS > 0)
   dump_history_enroll(Metal_r2, "<Metal_r2>");
#endif

  /* interface magnetic field */
#ifdef MHD
  generate();

  perturb(pDomain);
#endif  /* MHD */


  /* cell-centered magnetic field */
  /*   derive this from interface field to be internally consistent
       with athena */
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
#ifdef MHD
        pGrid->U[k][j][i].B1c =
          0.5 * (pGrid->B1i[k][j][i] + pGrid->B1i[k][j][i+1]);
        pGrid->U[k][j][i].B2c = pGrid->U[k][j][i].B3c = 0.0;

        if (pGrid->Nx[1] > 1)
          pGrid->U[k][j][i].B2c =
            0.5 * (pGrid->B2i[k][j][i] + pGrid->B2i[k][j+1][i]);
        if (pGrid->Nx[2] > 1)
          pGrid->U[k][j][i].B3c =
            0.5 * (pGrid->B3i[k][j][i] + pGrid->B3i[k+1][j][i]);

#ifndef ISOTHERMAL
        /* add magnetic energy to the total energy */
        pGrid->U[k][j][i].E +=
          0.5 * (SQR(pGrid->U[k][j][i].B1c)+SQR(pGrid->U[k][j][i].B2c)+
                 SQR(pGrid->U[k][j][i].B3c));
#endif  /* ISOTHERMAL */
#endif  /* MHD */
      }
    }
  }

  return;
}


/* use in problem() and read_restart()*/
static void set_vars(Real time)
{
  char *atm_file;
  Real dx, dxmin;
#ifdef STATIC_MESH_REFINEMENT
  int ir, irefine, nlevels;

#ifdef MPI_PARALLEL
  Real my_dxmin;
  int ierr;
#endif  /* MPI_PARALLEL */
#endif  /* STATIC_MESH_REFINEMENT */

  int iseed;

  iseed = -10;
#ifdef MPI_PARALLEL
  iseed -= myID_Comm_world;
#endif  /* MPI_PARALLEL */
  srand(iseed);

  /* Transport Coefficients */
#ifdef VISCOSITY
  NuFun_i = NULL;
  NuFun_a = nu_fun;
#endif  /* VISCOSITY */

#ifdef THERMAL_CONDUCTION
  KappaFun_i  = NULL;
  KappaFun_a  = kappa_fun;
  f_sp        = par_getd("problem", "f_sp");
#endif  /* THERMAL_CONDUCTION */

  atm_file = par_gets("problem", "atm_file");
  import_atmosphere(atm_file);

  xi_init = par_getd("problem", "xi_init");
  Tigm    = par_getd("problem", "Tigm");

  c_nfw = par_getd("problem", "c_nfw");
  f = 1.0/(log(1.0+c_nfw)/c_nfw - 1.0/(1.0+c_nfw));

  ExternalGravPot = grav_pot;

  rout = 0.9 * par_getd("domain1", "x1max");

  /* calculate dxmin.  this is pretty bad code... */
  dx = par_getd("domain1", "x1max")-par_getd("domain1", "x1min");
  dx /= (Real) par_geti("domain1", "Nx1");
  dxmin = dx;

  if (par_geti("domain1", "Nx2")>1) {
    dx = (par_getd("domain1", "x2max")-par_getd("domain1", "x2min"));
    dx /= (Real) par_geti("domain1", "Nx2");
    dxmin = MIN(dxmin, dx);
  }

  if (par_geti("domain1", "Nx3")>1) {
    dx = (par_getd("domain1", "x3max")-par_getd("domain1", "x3min"));
    dx /= (Real) par_geti("domain1", "Nx3");
    dxmin = MIN(dxmin, dx);
  }

#ifdef STATIC_MESH_REFINEMENT
  irefine = 1;
  nlevels = par_geti("job", "num_domains");
  for (ir=1; ir<nlevels; ir++) irefine *= 2;
  dxmin /= (Real) irefine;

  /* sync over all processors */
#ifdef MPI_PARALLEL
  my_dxmin = dxmin;
  ierr = MPI_Allreduce(&my_dxmin, &dxmin, 1,
                       MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif /* MPI_PARALLEL */
#endif  /* STATIC_MESH_REFINEMENT */



  rsoft = 4.0 * dxmin;

  t_ofst = par_getd("problem", "t_ofst");

  m15 = par_getd("problem", "M15"); /* M0 / 10^15 M_sun */

  cooling = par_geti_def("problem", "cooling", 0);

  /* needed for grav_pot */
  z = zz(time + t_ofst);
  m = mm(z);
  h = hh(z);

  rvir = pow(m, 1.0/3.0) * pow(10.0*h, -2.0/3.0);
  if (2.0 * rvir >= rout){
    ath_error("[Userwork_in_loop]: virial radius exceeds domain size.\n");
  }

  return;
}
/* end problem() */
/* ========================================================================== */


/* ========================================================================== */
/* cooling */

/* Given P and d in dimensionless units, return kT in keV. */
/* -- dimensionless units are such that G = M0 = H0 = 1. */
static Real kT_keV(const Real P, const Real d)
{
  return(4.690 * pow(m15,2.0/3.0) * mu * P / d);
}

/* Given T in keV, return Lambda / 1e-23 erg cm^3/s.  This is the fit
   from Tozzi & Norman 2001 and should probably be improved. */
static Real L_23(const Real T)
{
  return(0.63 + 0.58 * sqrt(T) + 0.086 * pow(T,-1.7));
}

static Real cool(const Real d, const Real P, const Real dt)
{
  Real kT, Edot;

  kT = kT_keV(P, d);
  Edot = 0.02704 * SQR(d*mue/mu) * pow(m15, -2.0/3.0) * L_23(kT);

  Edot *= mytanh(50.0 * (P/d - 0.15));

  return Edot;
}

static void cool_step(GridS *pGrid, const Real dt)
{
  int i, j, k;
  Real KE, ME, d, P, T;
  int is,ie,js,je,ks,ke;
  Real x1,x2,x3,r;

  is = pGrid->is; ie = pGrid->ie;
  js = pGrid->js; je = pGrid->je;
  ks = pGrid->ks; ke = pGrid->ke;

  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        cc_pos(pGrid,i,j,k,&x1,&x2,&x3);
        r = sqrt(x1*x1+x2*x2+x3*x3);

        if (r <= 2.0*rvir) {
          d = pGrid->U[k][j][i].d;

          KE = (SQR(pGrid->U[k][j][i].M1) +
                SQR(pGrid->U[k][j][i].M2) +
                SQR(pGrid->U[k][j][i].M3)) / (2.0 * d);

          ME = 0.0;
#ifdef MHD
          ME = (SQR(pGrid->U[k][j][i].B1c) +
                SQR(pGrid->U[k][j][i].B2c) +
                SQR(pGrid->U[k][j][i].B3c)) * 0.5;
#endif  /* MHD */

          P = (pGrid->U[k][j][i].E - ME - KE) * Gamma_1;

          pGrid->U[k][j][i].E -= dt * cool(d,P,dt);

          /* put in a temperature floor */
          /*  note: the floor here should be consistent with the tanh
              cutoff in cool() above.  if you change either, plot these
              in mathematica and make sure they work together. */
          T = Gamma_1 * (pGrid->U[k][j][i].E - KE - ME) / d;
          T = MAX(T, 0.1);

          pGrid->U[k][j][i].E = d*T/Gamma_1 + KE + ME;
        }
      }
    }
  }

  return;
}

static void integrate_cool(DomainS *pDomain, const Real dt_hydro)
{
  GridS *pGrid = pDomain->Grid;
  Real my_tcool, my_dtmin, dtmin;
  Real d, P, ME, KE;
  int ierr, n_sub;

  int i,j,k;
  int is,ie,js,je,ks,ke;

  is = pGrid->is; ie = pGrid->ie;
  js = pGrid->js; je = pGrid->je;
  ks = pGrid->ks; ke = pGrid->ke;

  /* find the shortest cooling time on the grid */
  my_dtmin = dt_hydro;
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        d = pGrid->U[k][j][i].d;

        KE = (SQR(pGrid->U[k][j][i].M1) +
              SQR(pGrid->U[k][j][i].M2) +
              SQR(pGrid->U[k][j][i].M3)) / (2.0 * d);

        ME = 0.0;
#ifdef MHD
        ME = (SQR(pGrid->U[k][j][i].B1c) +
              SQR(pGrid->U[k][j][i].B2c) +
              SQR(pGrid->U[k][j][i].B3c)) * 0.5;
#endif  /* MHD */

        P = (pGrid->U[k][j][i].E - ME - KE) * Gamma_1;


        my_tcool = P / Gamma_1 / cool(d, P, 0.0); /* use dt=0 in cool() */
        my_dtmin = MIN(my_dtmin, 0.25 * my_tcool); /* CFL = 0.25? */
      }
    }
  }

  /* sync over all processors */
#ifdef MPI_PARALLEL
  ierr = MPI_Allreduce(&my_dtmin, &dtmin, 1,
                       MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#else
  dtmin = my_dtmin;
#endif /* MPI_PARALLEL */


  /* run the subcycles */
  n_sub = (int) ceil(dt_hydro / dtmin);
  dtmin = dt_hydro / n_sub;
  ath_pout(0, "[integrate_cool]: running %d subcycles.\n", n_sub);

  for (i=0; i<n_sub; i++) {
    cool_step(pGrid, dtmin);
  }

  return;
}
/* ========================================================================== */


/* ========================================================================== */
/* import_atmosphere() */
static void import_atmosphere(char *atm_file)
/* Imports the supplied atmosphere file.  The file must be in the
   following format:

   # header information...
   # ...
   g = (McBride et al gamma)
   b = (McBride et al beta/gamma)
   n = (number of entries)

   x_i  v_i  rho_i
   ...

   where x = r/rvir and the data span at least from x=1.5 to x=2.0.
   also, there should be at least n
   entries. */
{
  FILE *input;
  char buff[512];
  Real x, v, rho;
  char c1, c2;
  int j;

  if((input = fopen(atm_file, "r")) == NULL)
    ath_error("[import_atmosphere]: Could not read file: %s\n", atm_file);

  /* Read lines from the file until we encounter n = %d */
  n_entries = 0;
    while (fgets(buff, sizeof(buff), input) != NULL) {
    if ((sscanf(buff, "%c %c %d", &c1, &c2, &n_entries) == 3)
        && (c1 == 'n') && (c2 == '=')){
      ath_pout(0, "[import_atmosphere]: Reading %d lines from input file.\n",
               n_entries);
      break;
    }
  }
  rewind(input);

  /* Read lines from the file until we encounter g = %f */
  while (fgets(buff, sizeof(buff), input) != NULL) {
    if ((sscanf(buff, "%c %c %le", &c1, &c2, &g) == 3)
        && (c1 == 'g') && (c2 == '=')){
      ath_pout(0, "[import_atmosphere]: parameter g = %f\n", g);
      break;
    }
  }
  rewind(input);

  /* Read lines from the file until we encounter b = %f */
  while (fgets(buff, sizeof(buff), input) != NULL) {
    if ((sscanf(buff, "%c %c %le", &c1, &c2, &b) == 3)
        && (c1 == 'b') && (c2 == '=')){
      ath_pout(0, "[import_atmosphere]: parameter b = %f\n", b);
      break;
    }
  }
  rewind(input);


  x_init = (Real*)calloc_1d_array(n_entries, sizeof(Real));
  v_init = (Real*)calloc_1d_array(n_entries, sizeof(Real));
  d_init = (Real*)calloc_1d_array(n_entries, sizeof(Real));

  j=0;
  while(fgets(buff, sizeof(buff), input) != NULL){
    if (sscanf(buff, "%le %le %le", &x, &v, &rho) == 3){
      if (j < n_entries){
        x_init[j] = x;
        v_init[j] = v;
        d_init[j] = rho;
        j++;
      }
    }
  }

  if (j < n_entries)
    ath_error("[import_atmosphere]: not enough entries in input file.\n  Expected %d, found %d.\n",
              n_entries, j);

  fclose(input);
}
/* end import_atmosphere() */
/* ========================================================================== */



/* ========================================================================== */
/* Cosmology */

/* Take the cosmic time t [in units of H0^(-1)] and return the
   corresponding redshift. */
static Real zz(Real t)
{
  Real a3;

  a3 = (Om/OL) * SQR(sinh(1.5 * sqrt(OL) * t));

  return (pow(a3, -1.0/3.0) - 1.0);
}

/* Given the redshift z1, return the normalized halo mass. */
static Real mm(Real z1)
{
  return pow(pow(1.0+z1, b) * exp(-z1), g);
}

/* Given the redshift z1, return the normalized Hubble parameter. */
static Real hh(Real z1)
{
  return (sqrt(OL + Om * pow(1.0+z1, 3.0)));
}

/* NFW potential. */
static Real phi_nfw(const Real r)
{
  Real x = r / rvir;
  return (-1.0 * f * log(1 + c_nfw*x)/(c_nfw*x) * pow(10.0*h*m, 2.0/3.0));
}
/* ========================================================================== */



/*==============================================================================
 * PROBLEM USER FUNCTIONS:
 * problem_write_restart() - writes problem-specific user data to restart files
 * problem_read_restart()  - reads problem-specific user data from restart files
 * get_usr_expr()          - sets pointer to expression for special output data
 * get_usr_out_fun()       - returns a user defined output function pointer
 * get_usr_par_prop()      - returns a user defined particle selection function
 * Userwork_in_loop        - problem specific work IN     main loop
 * Userwork_after_loop     - problem specific work AFTER  main loop
 *----------------------------------------------------------------------------*/

void problem_write_restart(MeshS *pM, FILE *fp)
{
  /*We're writing the address in memory of num, t and dt? Why?*/
  fwrite(&profile_dump.num, sizeof(int),   1, fp);
  fwrite(&profile_dump.t,   sizeof(Real),  1, fp);
  fwrite(&profile_dump.dt,  sizeof(Real),  1, fp);
  /*Do I still need return?*/
  return;
}

void problem_read_restart(MeshS *pM, FILE *fp)
{
  int prof_index,bin_index;

  set_vars(pM->time);

  /*Not sure what this does... not too worried about it*/
  profile_dump.n      = 100;
  profile_dump.out    = "prim";
  profile_dump.nlevel = -1;       /* dump all levels */

  fread(&profile_dump.num, sizeof(int),   1, fp);
  fread(&profile_dump.t,   sizeof(Real),  1, fp);
  fread(&profile_dump.dt,  sizeof(Real),  1, fp);

  /*Allocate and initialize array to hold profile data*/
  profile_data = (Real**)calloc_2d_array(n_bins, n_profiles, sizeof(Real));
  for(prof_index = 0; prof_index<n_profiles; prof_index++){
     for(bin_index = 0; bin_index<n_bins; bin_index++){
        profile_data[prof_index][bin_index] = 0.0;
     }
  }
#ifdef MPI_PARALLEL
  profile_data_global = (Real**) calloc_2d_array(n_bins, n_profiles, sizeof(Real));
  for(prof_index = 0; prof_index<n_profiles; prof_index++){
     for(bin_index = 0; bin_index<n_bins; bin_index++){
        profile_data_global[prof_index][bin_index] = 0.0;
     }
  }
#endif /*MPI_PARALLEL*/

  /* Free Athena-style arrays */
  free_3d_array(A1);
  free_3d_array(A2);
  free_3d_array(A3);

  /* Free FFTW-style arrays */
  ath_3d_fft_free(fA1);
  ath_3d_fft_free(fA2);
  ath_3d_fft_free(fA3);
  return;
}


#if (NSCALARS > 0)
static Real Metal_r2(const GridS *pG, const int i, const int j, const int k)
{
   Real x1,x2,x3,rsqr;
   cc_pos(pG,i,j,k,&x1,&x2,&x3);
   rsqr = x1*x1 + x2*x2 + x3*x3;
   return pG->U[k][j][i].s[0]*rsqr;
}

static Real metals(const GridS *pG, const int i, const int j, const int k)
{
   return pG->U[k][j][i].s[0]/pG->U[k][j][i].d;
}

static Real scalar1(const GridS *pG, const int i, const int j, const int k)
{
   return pG->U[k][j][i].s[1]/pG->U[k][j][i].d;
}

static Real scalar2(const GridS *pG, const int i, const int j, const int k)
{
   return pG->U[k][j][i].s[2]/pG->U[k][j][i].d;
}

static Real scalar3(const GridS *pG, const int i, const int j, const int k)
{
   return pG->U[k][j][i].s[3]/pG->U[k][j][i].d;
}

static Real scalar4(const GridS *pG, const int i, const int j, const int k)
{
   return pG->U[k][j][i].s[4]/pG->U[k][j][i].d;
}

static Real scalar5(const GridS *pG, const int i, const int j, const int k)
{
   return pG->U[k][j][i].s[5]/pG->U[k][j][i].d;
}
#endif

ConsFun_t get_usr_expr(const char *expr)
{
#if (NSCALARS > 0)
   if(strcmp(expr,"s1")==0) return scalar1;
   else if(strcmp(expr,"s2")==0) return scalar2;
   else if(strcmp(expr,"s3")==0) return scalar3;
   else if(strcmp(expr,"s4")==0) return scalar4;
   else if(strcmp(expr,"s5")==0) return scalar5;
   else if(strcmp(expr, "metals")==0) return metals;
#endif
  return NULL;
}

VOutFun_t get_usr_out_fun(const char *name){
  return NULL;
}

/* "inner" bc: set density, temperature, and velocity at r = 2 r_vir. */
/*   use global variables rho_out, Tigm, and rvir */
static void inner_bc(DomainS *pDomain)
{
  GridS *pGrid = pDomain->Grid;
  int i=0,j=0,k=0;
  int is,ie,js,je,ks,ke,iprob;
  Real x1,x2,x3, r;
#ifndef ISOTHERMAL
  Real KE, T, d, ME;
#endif  /* ISOTHERMAL */

  is = pGrid->is; ie = pGrid->ie;
  js = pGrid->js; je = pGrid->je;
  ks = pGrid->ks; ke = pGrid->ke;

  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        cc_pos(pGrid,i,j,k,&x1,&x2,&x3);
        r = sqrt(x1*x1+x2*x2+x3*x3);

        ME = 0.0;
#ifdef MHD
        ME = (SQR(pGrid->U[k][j][i].B1c) +
              SQR(pGrid->U[k][j][i].B2c) +
              SQR(pGrid->U[k][j][i].B3c)) * 0.5;
#endif  /* MHD */

        if (r >= 2.0*rvir) {
          pGrid->U[k][j][i].d = rho_out;

          /* seed a perturbation for the MTI */
#ifdef THERMAL_CONDUCTION
          if (r <= 2.1 * rvir)
            pGrid->U[k][j][i].d *= (1.0 + RandomNormal(0.0, 0.05));
#endif  /* THERMAL_CONDUCTION */

          pGrid->U[k][j][i].M1 = 0.0;
          pGrid->U[k][j][i].M2 = 0.0;
          pGrid->U[k][j][i].M3 = 0.0;

#ifndef ISOTHERMAL
          pGrid->U[k][j][i].E = (Tigm * rho_out)/Gamma_1;
          pGrid->U[k][j][i].E += ME;
#endif  /* ISOTHERMAL */
        }
      }
    }
  }

  return;
}

/*Function to make radial profiles within loop*/
/*didnt include int *nmax as argument to function*/
/*Already loop over domains in Userwork_in_loop so I wrote this to only get passed domain, call it within loop in Userwork*/
static void calc_profiles(DomainS *pDomain, Real **profile_data)
{
   int nl, nd;
   GridS *pGrid = pDomain->Grid;
   int is, ie, js, je, ks, ke;
   int i, j, k, s;
   double r;

   int prof_index;

   int profile_index=0,bin_index=0;

   Real x1, x2, x3, dx1;

   dx1 = pDomain->dx[1];


#ifdef MPI_PARALLEL
   int ierr;
#endif /*MPI_PARALLEL*/

   PrimS W;
   ConsS U;

   is = pGrid->is; ie=pGrid->ie;
   js = pGrid->js; je=pGrid->je;
   ks = pGrid->ks; ke=pGrid->ke;


   for(prof_index = 0; prof_index<n_profiles; prof_index++){
      for(bin_index = 0; bin_index<n_bins; bin_index++){
         profile_data[prof_index][bin_index] = 0.0;
      }
   }

#ifdef MPI_PARALLEL
   for(prof_index = 0; prof_index<n_profiles; prof_index++){
      for(bin_index = 0; bin_index<n_bins; bin_index++){
         profile_data_global[prof_index][bin_index] = 0.0;
      }
   }
#endif /*MPI_PARALLEL*/


   for(k=ks; k<=ke; k++){
      for(j=js; j<=je; j++){
         for(i=is; i<=ie; i++){
           /*calculate radius in cell coordinates and corresponding index*/
           cc_pos(pGrid,i,j,k,&x1,&x2,&x3);
           r = sqrt(x1*x1 + x2*x2 + x3*x3);

            /*FIX ME: Make binning more accurate*/
            s = (int) floor(r/dx1);

            W = Cons_to_Prim(&(pGrid->U[k][j][i]));

            /*0 contains number of bins correspinding to a given s*/
            profile_data[0][s] += 1.0;

            /*1 sums radii so that we can calculate avg R corresponding to a given s*/
            profile_data[1][s] += r;

            /*density*/
            profile_data[2][s] += W.d;

            /*pressure*/
            profile_data[3][s] += W.P;

            /*temperature*/
            profile_data[4][s] += W.P/W.d;

            /*Mach number squared*/
            profile_data[5][s] += W.d*(SQR(W.V1) + SQR(W.V2) + SQR(W.V3)) / W.P;

            /*Entropy*/
            profile_data[6][s] += W.P / pow(W.d,5.0/3.0);

	    /*Radial Velocity*/
	    profile_data[7][s] += (W.V1*x1 + W.V2*x2 + W.V3*x3)/r;

	    /*Total Metals in a shell*/
	    profile_data[8][s] += pGrid->U[k][j][i].s[0]*4.0*PI*SQR(r);
#ifdef MHD
            /*B^2*/
            profile_data[10][s] += SQR(W.B1c) + SQR(W.B2c) + SQR(W.B3c);

            /*beta*/
            profile_data[11][s] += 2.0*W.P/(SQR(W.B1c) + SQR(W.B2c) +
            SQR(W.B3c));

            /*Br*/
            profile_data[12][s] += (W.B1c*x1 + W.B2c*x2 + W.B3c*x3)/r;

	    /*Alfven Mach Number squared*/
	    profile_data[13][s] += W.d*(SQR(W.V1) + SQR(W.V2) + SQR(W.V3))/(SQR(W.B1c) + SQR(W.B2c) + SQR(W.B3c));
#endif /*MHD*/
         }
      }
   }


/*Note that since prof_data[9][*] is still 0, so too will prof_data_global[9][*]*/
#ifdef MPI_PARALLEL
   ierr = MPI_Allreduce(&profile_data[0][0], &profile_data_global[0][0], n_bins*n_profiles,
                        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   if(ierr)
      ath_error("[calc_profiles]: MPI_Allreduce returned error %d\n", ierr);

   for(profile_index=0; profile_index<n_profiles; profile_index++){
      for(bin_index=0; bin_index<n_bins; bin_index++){
         profile_data[profile_index][bin_index] = profile_data_global[profile_index][bin_index];
      }
   }
#endif /*MPI_PARALLEL*/



   /*Divide through by 0th row of array (num) to get proper averages*/
   for(profile_index=1; profile_index<n_profiles; profile_index++){
     for(bin_index=0; bin_index<n_bins; bin_index++){
        if(profile_data[0][bin_index]!= 0.0){
          profile_data[profile_index][bin_index] /= profile_data[0][bin_index];
        }
     }
   }

   /*Now that we have averages of T and Vr on the shell, go back and calculate convective heat fluxes*/
   for(k=ks; k<=ke; k++){
      for(j=js; j<=je; j++){
         for(i=is; i<=ie; i++){
           /*calculate radius in cell coordinates and corresponding index*/
           cc_pos(pGrid,i,j,k,&x1,&x2,&x3);
           r = sqrt(x1*x1 + x2*x2 + x3*x3);

            /*FIX ME: Make binning more accurate*/
            s = (int) floor(r/dx1);

            W = Cons_to_Prim(&(pGrid->U[k][j][i]));

	    /*(T-<T>)(Vr-<Vr>)*/
	    profile_data[9][s] += (W.P/W.d - profile_data[4][s])*((W.V1*x1 + W.V2*x2 + W.V3*x3)/r - profile_data[7][s]);
         }
      }
   }

/*reduce arrays into prof_data_global, but only copy prof_data_global[9][*] back*/
#ifdef MPI_PARALLEL
   ierr = MPI_Allreduce(&profile_data[9][0], &profile_data_global[9][0], n_bins,
                        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   if(ierr)
      ath_error("[calc_profiles]: MPI_Allreduce returned error %d\n", ierr);

   for(bin_index=0; bin_index<n_bins; bin_index++){
      profile_data[9][bin_index] = profile_data_global[9][bin_index];
   }
#endif /*MPI_PARALLEL*/

   /*Divide through by 0th row of array (num) to get proper averages, but again only for prof_data[9][*]*/
   for(bin_index=0; bin_index<n_bins; bin_index++){
     if(profile_data[0][bin_index]!= 0.0){
       profile_data[9][bin_index] /= profile_data[0][bin_index];
     }
   }


   return;

}


void dump_profile(DomainS *pD, OutputS *pOut)
{
  FILE *pfile;
  char *fname,*plev=NULL,*pdom=NULL;
  char levstr[8],domstr[8];
  Real x1,x2,x3;
  char zone_fmt[20], fmt[80];
  int col_cnt;
  GridS *pGrid = pD->Grid;
  char outfilename[80] = "cluster";

  /* Anoop, add whatever variables you need here */
  int c;

  /* Anoop, mostly ignore this stuff until you see things flagged TODO */

  sprintf(fmt," %%12.8e"); /* Use a default format */

  col_cnt = 1;

  /* construct output filename. */
  if((fname = ath_fname(NULL,
                        outfilename,
                        NULL,
                        NULL,
                        num_digit,
                        pOut->num,
                        NULL,
                        "pro")) == NULL){
    ath_error("[dump_profile]: Error constructing filename\n");
  }

  /* open output file */
  if((pfile = fopen(fname,"w")) == NULL){
    ath_error("[dump_profile]: Unable to open text file %s\n",fname);
  }
  free(fname);

  /* Upper and Lower bounds on i,j,k for data dump */
  sprintf(zone_fmt,"%%%dd", (int)(2+log10((double)(n_bins))));

  /* Write out some header information */
  if (pGrid->Nx[0] > 1) {
    fprintf(pfile,"# N = %d\n", n_bins);
  }
  fprintf(pfile,"# RADIAL PROFILE at Time= %g\n", pGrid->time);

  /* write out column headers.  Note column number is embedded in header */
  fprintf(pfile,"# [%d]=i",col_cnt);
  col_cnt++;

  if (pGrid->Nx[0] > 1) {
    fprintf(pfile," [%d]=r",col_cnt);
    col_cnt++;
  }

  /* TODO: change to the vars you want */
  /* MM recommends: density, pressure, temperature, entropy, mach
     number, maybe radial velocity.  try to keep this code flexible so
     you can change it */
  /* later, let's add magnetic energy and plasma beta */

  fprintf(pfile," [%d]=rho",col_cnt);
  col_cnt++;
  fprintf(pfile," [%d]=P",col_cnt);
  col_cnt++;
  fprintf(pfile," [%d]=T",col_cnt);
  col_cnt++;
  fprintf(pfile," [%d]=M^2",col_cnt);
  col_cnt++;
  fprintf(pfile," [%d]=K",col_cnt);
  col_cnt++;
  fprintf(pfile," [%d]=Vr",col_cnt);
  col_cnt++;
  fprintf(pfile," [%d]=Metal",col_cnt);
  col_cnt++;
  fprintf(pfile," [%d]=Convective Flux",col_cnt);
  col_cnt++;
#ifdef MHD
  fprintf(pfile," [%d]=B^2",col_cnt);
  col_cnt++;
  fprintf(pfile," [%d]=Beta",col_cnt);
  col_cnt++;
  fprintf(pfile," [%d]=Br",col_cnt);
  col_cnt++;
  fprintf(pfile," [%d]=Ma^2",col_cnt);
  col_cnt++;
#endif /*MHD*/

  fprintf(pfile,"\n");

  /* Write out data */

  for(c=0; c<n_bins; c++){
    fprintf(pfile, zone_fmt, c);
    fprintf(pfile, fmt, profile_data[1][c]);

    /* Dump all variables */
    fprintf(pfile, fmt, profile_data[2][c]);
    fprintf(pfile, fmt, profile_data[3][c]);
    fprintf(pfile, fmt, profile_data[4][c]);
    fprintf(pfile, fmt, profile_data[5][c]);
    fprintf(pfile, fmt, profile_data[6][c]);
    fprintf(pfile, fmt, profile_data[7][c]);
    fprintf(pfile, fmt, profile_data[8][c]);
    fprintf(pfile, fmt, profile_data[9][c]);
#ifdef MHD
    fprintf(pfile, fmt, profile_data[10][c]);
    fprintf(pfile, fmt, profile_data[11][c]);
    fprintf(pfile, fmt, profile_data[12][c]);
    fprintf(pfile, fmt, profile_data[13][c]);
#endif /*MHD*/

    fprintf(pfile,"\n");
  }

  fclose(pfile);

  return;
}


/* calculate rho and T at the turnaround radius and call inner_bc() */
void Userwork_in_loop(MeshS *pM)
{
  int nl, nd;
#ifdef MPI_PARALLEL
  int ierr;
#endif	/* MPI_PARALLEL */

  z = zz(pM->time + t_ofst);
  m = mm(z);
  h = hh(z);

  rvir = pow(m, 1.0/3.0) * pow(10.0*h, -2.0/3.0);
  if (2.0 * rvir >= rout){
    ath_error("[Userwork_in_loop]: virial radius exceeds domain size.\n");
  }

  ath_pout(0, "[z  m  h  rv] = [%f  %f  %f  %f]\n", z, m, h, rvir);

  /* construct the outer density */
  /* -- start with the critical density at rvir */
  rho_out = 200.0 * fb * 3.0/(8*PI) * SQR(h);
  /* -- scale to 2*r_vir using nfw profile */
  rho_out *=  log(1.0+2.0*c_nfw)/log(1.0+c_nfw)/8.0;
  /* -- fix to the accretion rate (unique to James' formula)*/
  rho_out /= (1.0 + 3.0*Om/g * pow(1.0+z, 3)/(1.0+z-b)/SQR(h));

  for (nl=0; nl<=(pM->NLevels)-1; nl++) {
    for (nd=0; nd<=(pM->DomainsPerLevel[nl])-1; nd++) {
      if (pM->Domain[nl][nd].Grid != NULL) {
        inner_bc(&(pM->Domain[nl][nd]));

         /* check whether we need to do an output */
         if(pM->time >= profile_dump.t){

            /* first, update output time */
            profile_dump.t += profile_dump.dt;

            /* next, calculate the radial profiles and store them in the global array */
            calc_profiles(&(pM->Domain[nl][nd]), profile_data);

            /* finally, write the data to disk, but only on the root process */
            /*Fancy configuration of if and ifdefs, not sure why this is necessary*/
#ifdef MPI_PARALLEL
            if (myID_Comm_world == 0){
#endif /* MPI_PARALLEL */
               dump_profile(&(pM->Domain[nl][nd]), &profile_dump);
#ifdef MPI_PARALLEL
            }
#endif /* MPI_PARALLEL */

            profile_dump.num += 1;
         }
        if (cooling > 0)
          integrate_cool(&(pM->Domain[nl][nd]), pM->dt);
      }
    }
  }

  return;
}

void Userwork_after_loop(MeshS *pM)
{
  int nl, nd;
  for (nl=0; nl<=(pM->NLevels)-1; nl++) {
    for (nd=0; nd<=(pM->DomainsPerLevel[nl])-1; nd++) {
      if (pM->Domain[nl][nd].Grid != NULL) {
            calc_profiles(&(pM->Domain[nl][nd]), profile_data);

            /* finally, write the data to disk, but only on the root process */
#ifdef MPI_PARALLEL
            if (myID_Comm_world == 0){
#endif /* MPI_PARALLEL */
               dump_profile(&(pM->Domain[nl][nd]), &profile_dump);
#ifdef MPI_PARALLEL
            }
#endif /* MPI_PARALLEL */
                        }
                }
        }

  /* free memory if necessary */
  if (profile_data != NULL) free_2d_array((void**) profile_data);
  if (A1 != NULL) free_3d_array((void***) A1);
  if (A2 != NULL) free_3d_array((void***) A2);
  if (A3 != NULL) free_3d_array((void***) A3);

#ifdef MPI_PARALLEL
  if (profile_data_global != NULL) free_2d_array((void**) profile_data_global);
#endif /* MPI_PARALLEL */

  return;
}


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
#define NDIV (1+IMM1/NTAB)
#define RNMX (1.0-DBL_EPSILON)
#define NTAB 32

/*! \fn double ran2(long int *idum){
 *  \brief The routine ran2() is extracted from the Numerical Recipes in C
 *
 * The routine ran2() is extracted from the Numerical Recipes in C
 * (version 2) code.  I've modified it to use doubles instead of
 * floats. -- T. A. Gardiner -- Aug. 12, 2003
 *
 * Long period (> 2 x 10^{18}) random number generator of L'Ecuyer
 * with Bays-Durham shuffle and added safeguards.  Returns a uniform
 * random deviate between 0.0 and 1.0 (exclusive of the endpoint
 * values).  Call with idum = a negative integer to initialize;
 * thereafter, do not alter idum between successive deviates in a
 * sequence.  RNMX should appriximate the largest floating point value
 * that is less than 1. */

double ran2(long int *idum){
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

#undef OFST
#undef KCOMP
#undef KWVM
/* end userwork */
/* ========================================================================== */


/* ========================================================================== */
/* gravity */
static Real mytanh(const Real x)
{
  return 0.5 * (1.0 + tanh(x));
}

/* Use static variables M and rvir in the potential */
static Real grav_pot(const Real x1, const Real x2, const Real x3, const Real time)
{
  Real phi_tot, phi_in, phi_out;
  Real r, r_switch, width;

  r = sqrt(x1*x1+x2*x2+x3*x3);
  r_switch = 0.8*3.0*rvir;      /* should I be using 2.0 rvir? */

  /* if (r < rsoft) r = rsoft; */

  width = 10.0/r_switch;

  phi_in  = phi_nfw(r);
  phi_out = phi_nfw(r_switch);  /* phi = const at large radii */

  /* interpolate between phi_in and phi_out */
  phi_tot = phi_in  * mytanh(width * (r_switch-r)) +
            phi_out * mytanh(width * (r-r_switch));

  return phi_tot;
}
/* end gravity */
/* ========================================================================== */


/* ========================================================================== */
/* Transport Coefficients */
#ifdef THERMAL_CONDUCTION
static Real kappa_fun(const Real d, const Real T,
                      const Real x1, const Real x2, const Real x3)
{
  return (1.039 * f_sp * pow(mu, 3.5) / mue * m15 * pow(T, 2.5));
}
#endif  /* THERMAL_CONDUCTION */

#ifdef VISCOSITY
static Real nu_fun(const Real d, const Real T,
                   const Real x1, const Real x2, const Real x3)
{
  Real fact = 1.0 + 0.11 * sqrt(2);  /* add helium contribution */
  return (0.00950 * fact * (mue/mu) * kappa_fun(d,T,x1,x2,x3) / d);
}
#endif  /* VISCOSITY */
/* end transport coefficients */
/* ========================================================================== */
