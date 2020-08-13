#include "main.h"
#include "mpi.h"
/*********************** self documentation **********************/
char *sdoc[] = {
"									",
" SUMIGGBZO - MIGration via Gaussian Beams of Zero-Offset SU data	",
"									",
" revised based on the basic code of Seismic Unix

" version 1, Aug 13,2020,
NULL};
/**************** end self doc ***********************************/

/* Credits:
 *
 *	CWP: Dave (algorithm), Jack and John (reformatting for SU)
 */


/* Ray types */
/* one step along ray */

typedef struct RayStepStruct
{
	float t;		/* time */
	float a;		/* angle */
	float x,z;		/* x,z coordinates */
	float q1,p1,q2,p2;	/* Cerveny's dynamic ray tracing solution */
	int kmah;		/* KMAH index */
	float c,s;		/* cos(angle) and sin(angle) */
	float v,dvdx,dvdz;	/* velocity and its derivatives */
    float vpphase;
	float dvpphasedx,dvpphasedz,ddvpphasedxdx,ddvpphasedzdz,ddvpphasedxdz;
	float dvpgroupdx,dvpgroupdz;
	float vpgroup;
	/*float apgroup;*/
} RayStep;

/* one ray */
typedef struct RayStruct
{
	int nrs;		/* number of ray steps */
	RayStep *rs;		/* array[nrs] of ray steps */
	int nc;			/* number of circles */
	int ic;			/* index of circle containing nearest step */
	void *c;		/* array[nc] of circles */
} Ray;

/* size of cells in which to linearly interpolate complex time and amplitude */
#define CELLSIZE 6

/* factor by which to oversample time for linear interpolation of traces */
#define NOVERSAMPLE 4

/* number of exponential decay filters */
#define NFILTER 10

/* exp(EXPMIN) is assumed to be negligible */
#define EXPMIN (-5.0)

/* filtered complex beam data as a function of real and imaginary time */
typedef struct BeamDataStruct
{
	int ntr;		/* number of real time samples */
	float dtr;		/* real time sampling interval */
	float ftr;		/* first real time sample */
	int nti;		/* number of imaginary time samples */
	float dti;		/* imaginary time sampling interval */
	float fti;		/* first imaginary time sample */
	complex ***cf;		/* array[npx][nti][ntr] of complex data */
} BeamData;

/* one cell in which to linearly interpolate complex time and amplitude */
typedef struct CellStruct
{
	int live;	/* random number used to denote a live cell */
	int ip;	/* parameter used to denote ray parameter */
	float tr;	/* real part of traveltime */
	float ti;	/* imaginary part of traveltime */
	float ar;	/* real part of amplitude */
	float ai;	/* imaginary part of amplitude */
	float angle;  /*angle of beam*/
} Cell;

/* structure containing information used to set and fill cells */
typedef struct CellsStruct
{
	int nt;		/* number of time samples */
	float dt;	/* time sampling interval */
	float ft;	/* first time sample */
	int lx;		/* number of x samples per cell */
	int mx;		/* number of x cells */
	int nx;		/* number of x samples */
	float dx;	/* x sampling interval */
	float fx;	/* first x sample */
	int lz;		/* number of z samples per cell */
	int mz;		/* number of z cells */
	int nz;		/* number of z samples */
	float dz;	/* z sampling interval */
	float fz;	/* first z sample */
	int live;	/* random number used to denote a live cell */
	int ip;	    /* parameter used to denote ray parameter */
	float wmin;	/* minimum (reference) frequency */
	float lmin;	/* minimum beamwidth for frequency wmin */
	Cell **cell; /* cell array[mx][mz] */
	Ray *ray;	/* ray */
//	BeamData *bd;	/* complex beam data as a function of complex time */
//	float **g;	/* array[nx][nz] containing g(x,z) */
} Cells;

/* Input the shot gather and head info*/

void inputrace(int is, int nt, float dx, int maxtr, FILE *fp, int *sisp, int *firisp, int *nistr);

/* Ray functions */
Ray *makeRay (float x0, float z0, float a0, int nt, float dt, float ft,
	int nx, float dx, float fx, int nz, float dz, float fz, float **v,
	float **del,float **ea,float **vpphase,float **vpgroup);
void freeRay (Ray *ray);
int nearestRayStep (Ray *ray, float x, float z);

/* Velocity functions */
/*void* vel2Alloc (int nx, float dx, float fx,
	int nz, float dz, float fz, float **v);
	*/

void* vel2Alloc (int nx, float dx, float fx,int nz, float dz, float fz, float **v,float **del,float **ea);

void* vel3Alloc (int nx, float dx, float fx,int nz, float dz, float fz, float **vpphase);
void* vel4Alloc (int nx, float dx, float fx,int nz, float dz, float fz, float **vpgroup);
void vel2Free (void *vel2);
void vel3Free (void *vel3);
void vel4Free (void *vel4);

void vel2Interp (void *vel2, float x, float z,
	float *v, float *vx, float *vz, float *vxx, float *vxz, float *vzz,float *del,float *ea);
void vel3Interp (void *vel3, float x, float z, float *vpphasex, float *vpphasez, float *vpphasexx, float *vpphasezz,float *vpphasexz);


void vel4Interp (void *vel4, float x, float z, float *vpgroupx, float *vpgroupz);

/* Beam functions */
void formBeams (float bwh, float dxb, float fmin,
	int nt, float dt, float ft,
	int nx, float dx, float fx, float **f,
	int ntau, float dtau, float ftau,
	int npx, float dpx, float fpx, float **g);
void accray (Ray *ray,Cell **c,float fmin, float lmin, int lx, int lz, int mx,
	int mz, int live, int ipx, int nt, float dt, float ft,
	int nx, float dx, float fx, int nz, float dz, float fz, float **g, float **v);
void scanimg(Cell ***c1, Cell ***c2, int live, int dead, int nt, float dt, float ft,
	float fmin, int lx, int lz, int nx, int nz, int mx, int mz, int npx, float **f, float**g, float ***adcig,
	float fpx, float dpx);

/* functions defined and used internally */
static void csmiggb(float bwh, float fmin, float fmax, float amin, float amax, int live,
	int dead, int nt, float dt, int sp,int firp, int nx, float dx, int ntr, float dtr, int nz, float dz,
	float **f, float **v, float **g, float ***adcig,float **del,float **ea,float **vpphase,float **vpgroup);
void zero1int(int *p,int n1);

void zero1float(float *p,int n1);

void zero1complex(complex *p,int n1);

void zero2int(int **p,int n1,int n2);

void zero2float(float **p,int n1,int n2);

void zero2complex(complex **p,int n1,int n2);

void zero3float(float ***p,int n1,int n2, int n3);

void zero3int(int ***p,int n1,int n2, int n3);

void zero3complex(complex ***p,int n1,int n2, int n3);



int main(int argc, char *argv[])
{
	int nx,nz,nt,ix,iz,ns,is,ntr,itr,sp,iangle,i,j,tracepad;
	int firp; /* first receive cdp of one shot gather*/
	float dx,dz,dt,fmin,fmax,amin,amax,vavg,bwh,dtr,
		**f,**g,**v;
	float **del,**ea;
	float ***adcig,***cig;
	char *vfile,*seifile,*imgfile;
    char *deltafile,*eatafile;
	int live,dead,maxtrace;
	int nangle;
	float ft;
	float **vpphase,**vpgroup;
	float apgroup;
    int np,myid;

	int nsbegin;

	FILE *vfp;
	FILE *sfp,*dfp;		/* temp file to hold traces	*/
    FILE *cigp;


	FILE *deltafp;
    FILE *eatafp;

//	FILE *tpfp;


	ft=1.0;
	nt = 4001;      /*time sampling*/
        vfile="velsmooth2.dat";
	deltafile="deltasmooth2.dat";
        eatafile="epsilonsmooth2.dat";

	seifile="shotfinal.dat";
	imgfile="image_1_15degree_1_250_dy_new.dat";
	ns=250;          /*shot number*/
//	ns=100;
//	ns=5;
	nsbegin=1;
//	ntr=601;
	ntr=1000;        /*trace number*/
	nz=550;         /*depth samples*/
    	dz=8.0*ft;         /*depth sampling interval*/
	dtr=10.0*ft;         /*trace interval*/
        dt = 0.0008;     /*time sampling interval*/
	dx = 10.0*ft;      /*cdp interval*/
	nx=1000;         /*cdp number*/
    	fmin = 10;
	fmax = 4.0*fmin;
    	amax = 50.0;
	amin = -amax;
	nangle=50;


       MPI_Init(&argc,&argv);
        MPI_Comm_size(MPI_COMM_WORLD,&np);
        MPI_Comm_rank(MPI_COMM_WORLD,&myid);

	vfp=fopen(vfile,"rb+");
	deltafp=fopen(deltafile,"rb+");
    eatafp=fopen(eatafile,"rb+");

	sfp=fopen(seifile,"rb+");
	dfp=fopen(imgfile,"wb+");
	cigp=fopen("adcig1_250.dat","wb");
//	tpfp=fopen("tpfp.dat","wb");


	/* random numbers used to denote live and dead cells */
	live = 1+(int)(1.0e7*franuni());
	dead = 1+(int)(1.0e7*franuni());
//	printf("%d",nangle);




	/* allocate workspace */

	adcig=alloc3float(nz,nangle,nx);
	v = alloc2float(nz,nx);
	vpphase =alloc2float(nz,nx);
	vpgroup =alloc2float(nz,nx);

	del=alloc2float(nz,nx);
    ea=alloc2float(nz,nx);
	g = alloc2float(nz,nx);
    cig=alloc3float(nz,nangle,nx);

	zero3float(adcig,nz,nangle,nx);
    zero3float(cig,nz,nangle,nx);
	zero2float(v,nz,nx);
	zero2float(g,nz,nx);
	zero2float(vpphase,nz,nx);
	zero2float(vpgroup,nz,nx);
	zero2float(del,nz,nx);
	zero2float(ea,nz,nx);


	/* load traces into the zero-offset array and close tmpfile */

	for(ix=0;ix<nx;ix++)
	{
	fread(v[ix], sizeof(float), nz, vfp);
	}
	fclose(vfp);
/*	fclose(sfp);*/


	for(ix=0;ix<nx;ix++)
	{
	fread(del[ix], sizeof(float), nz, deltafp);
	}
	fclose(deltafp);

	for(ix=0;ix<nx;ix++)
	{
	fread(ea[ix], sizeof(float), nz, eatafp);
	}
	fclose(eatafp);





	for (ix=0,vavg=0.0; ix<nx; ++ix)
	{
		for (iz=0; iz<nz; ++iz)
		{
			v[ix][iz] *= 1.0*ft;

			vavg += v[ix][iz];
		}
	}
	vavg /= nx*nz;

	/* get beam half-width */
	bwh = vavg/fmin/1.0;
    printf("%f\n",bwh);


	/* zero migrated image */
	for (ix=0; ix<nx; ++ix)
	for (iz=0; iz<nz; ++iz)
		g[ix][iz] = 0.0;

	/* zero adcigs */
	for(i=0;i<nx;i++)
	    for(iangle=0;iangle<nangle;iangle++)
			for(j=0;j<nz;j++)
			{
		    	adcig[i][iangle][j]=0.0;

			}
        for(i=0;i<nx;i++)
            for(iangle=0;iangle<nangle;iangle++)
                        for(j=0;j<nz;j++)
                        {
                        cig[i][iangle][j]=0.0;

                        }
	/* loops over shot*/
           for(is=nsbegin+myid*1;is<=ns;is=is+1*np)
	{
	       // inputrace(is,nt,dx,maxtrace,sfp,&sp,&firp,&ntr);
			f=alloc2float(nt,ntr);
			zero2float(f,nt,ntr);
			sp=1+(is-1)*4;
		//	firp=sp-150;
			firp=1;

	     for(i=0;i<ntr;i++)
				for(j=0;j<nt;j++)
					f[i][j]=0.0;
                printf("is=%d,sp=%d,firp=%d\n",is,sp,firp);
				fseek(sfp,4*nt*ntr*(is-1),0);
			for(i=0;i<ntr;i++)
			{
			//	fseek(sfp,60*4,1);
                fread(f[i], sizeof(float), nt, sfp);
			}
/*	        for(ix=0;ix<ntr;ix++)
			{
              fwrite(f[ix],sizeof(float),nt,tpfp);
			}*/

			if(is<=ns)
			{
        	/* migrate */
	          csmiggb(bwh,fmin,fmax,amin,amax,live,dead,nt,dt,sp,firp,nx,dx,ntr,dtr,nz,dz,f,v,g,adcig,del,ea,vpphase,vpgroup);
			}
			free2float(f);

	}
        MPI_Reduce(&adcig[0][0][0],&cig[0][0][0],nx*nangle*nz,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        free3float(adcig);

       if(myid==0)
        {

	//Output Adcigs//
	for(i=0;i<nx;i++)
		for(iangle=0;iangle<nangle;iangle++)
		{
		fwrite(cig[i][iangle],sizeof(float),nz,cigp);
		}

    //stack Adcigs//
	for(i=0;i<nx;i++)
		for(j=0;j<nz;j++)
	    	for(iangle=0;iangle<nangle;iangle++)
			{
			//printf("iangle=%d\n",iangle);
				if(iangle<15||iangle>35)
				cig[i][iangle][j]=0.0;

				else
				cig[i][iangle][j]=cig[i][iangle][j];

				g[i][j]+=cig[i][iangle][j];
	//			printf("%f\n",adcig[i][iangle][j]);
			}


    //write out//
	for(ix=0;ix<nx;ix++)
	{
        fwrite(g[ix],sizeof(float),nz,dfp);
	}
        }
	/* free workspace */
	fclose(cigp);
	fclose(sfp);
	fclose(dfp);
//	fclose(deltafp);
//	fclose(eatafp);
//	free3float(adcig);
	free2float(v);
	free2float(g);
	free2float(vpphase);
	free2float(vpgroup);

	free2float(del);
	free2float(ea);
        MPI_Finalize();
	return EXIT_SUCCESS;
}


static void csmiggb (float bwh, float fmin, float fmax, float amin, float amax, int live,
	int dead, int nt, float dt, int sp,int firp, int nx, float dx, int ntr, float dtr, int nz, float dz,
	float **f, float **v, float **g, float ***adcig,float **del,float **ea,float **vpphase,float **vpgroup)
/*****************************************************************************
Migrate zero-offset data via accumulation of Gaussian beams.
******************************************************************************
      nput:
bwh		horizontal beam half-width at surface z=0
fmin		minimum frequency (cycles per unit time)
fmax		maximum frequency (cycles per unit time)
amin		minimum emergence angle at surface z=0 (degrees)
amax		maximum emergence angle at surface z=0 (degrees)
nt		number of time samples
dt		time sampling interval (first time assumed to be zero)
nx		number of x samples
dx		x sampling interval
ntr     trace samples
dtr     trace sampling interval
nz		number of z samples
dz		z sampling interval
f		array[nx][nt] containing zero-offset data f(t,x)
v		array[nx][nz] containing half-velocities v(x,z)

Output:
g		array[nx][nz] containing migrated image g(x,z)
*****************************************************************************/
{
	int nxb,npx,ntau,ipx,ix,ixb,ixlo,ixhi,nxw,iz,mx,mz,lx,lz,rayxp;
        int i;
			int jx,jz;
	float ft,fx,fz,xwh,dxb,fxb,xb,vmin,dpx,fpx,px,
		taupad,dtau,ftau,fxw,pxmin,pxmax,
		a0,x0,z0,bwhc,**b;
	Ray *ray1,*ray2;
	Cell ***cell1,***cell2;

	/* first t, x, and z assumed to be zero */
	ft = fx = fz = 0.0;

//  size of coarse gird:cell
	lx=CELLSIZE;
	lz=CELLSIZE;
//  number of cells
	mx=2+(nx-1)/lx;
	mz=2+(nz-1)/lz;

	/* convert minimum and maximum angles to radians */
	amin *= PI/180.0;
	amax *= PI/180.0;
	if (amin>amax)
	{
		float atemp=amin;
		amin = amax;
		amax = atemp;
	}

	/* window half-width */
	xwh = 3.0*bwh;

	/* beam center sampling */

	dxb = NINT((1.8*bwh*sqrt(fmin/fmax))/dtr)*dtr;

	nxb = 1+(ntr-1)*dtr/dxb;

	fxb = fx+0.5*((ntr-1)*dtr-(nxb-1)*dxb);


	fprintf(stderr,"nxb=%d dxb=%g fxb=%g\n",nxb,dxb,fxb);

	/* minimum velocity at surface z=0 */
	for (ix=1,vmin=v[0][0]; ix<nx; ++ix)
		if (v[ix][0]<vmin) vmin = v[ix][0];

	/* beam sampling */
	pxmin = sin(amin)/vmin;
	pxmax = sin(amax)/vmin;
//	dpx = 2.0/(2.0*xwh*sqrt(fmin*fmax));
	dpx=1.0/(8.0*bwh*sqrt(fmin*fmax));
	npx = 1+(pxmax-pxmin)/dpx;
	fpx = pxmin+0.5*(pxmax-pxmin-(npx-1)*dpx);
	taupad = MAX(ABS(pxmin),ABS(pxmax))*xwh;
	taupad = NINT(taupad/dt)*dt;
	ftau = ft-taupad;
	dtau = dt;
	ntau = nt+2.0*taupad/dtau;

	printf("live=%d,dead=%f\n",ntau,ftau);

	cell1=(Cell***)alloc3(mz,mx,npx,sizeof(Cell));
	for (ipx=0; ipx<npx; ++ipx)
        for(jx=0;jx<mx;jx++)
		    for(jz=0;jz<mz;jz++)
              {
				cell1[ipx][jx][jz].live=0;
				cell1[ipx][jx][jz].ip=0;
				cell1[ipx][jx][jz].tr=0.0;
				cell1[ipx][jx][jz].ti=0.0;
				cell1[ipx][jx][jz].ar=0.0;
				cell1[ipx][jx][jz].ai=0.0;
				cell1[ipx][jx][jz].angle=0.0;
			  }

	fprintf(stderr,"npx=%d dpx=%g fpx=%g\n",npx,dpx,fpx);



	/*rays form shot position*/
	for (ipx=0,px=fpx+0*dpx; ipx<npx; ++ipx,px+=dpx) {

		/* emergence angle and location */
		a0 = -asin(px*v[sp-1][0]);
		x0 = fx+(sp-1)*dx;
		z0 = fz;
     	if (px*v[sp-1][0]>sin(amax)+0.01) continue;
		if (px*v[sp-1][0]<sin(amin)-0.01) continue;


		/* beam half-width adjusted for cosine of angle */
		bwhc = bwh*cos(a0);

		/* trace ray */
		ray1 = makeRay(x0,z0,a0,nt,dt,ft,nx,dx,fx,nz,dz,fz,v,del,ea,vpphase,vpgroup);
		accray(ray1,cell1[ipx],fmin,bwhc,lx,lz,mx,mz,live,ipx,ntau,dtau,ftau,
				nx,dx,fx,nz,dz,fz,g,v);
		freeRay(ray1);
	}




		/* loop over beam centers */
	for (ixb=0,xb=fxb+0*dxb; ixb<nxb; ++ixb,xb+=dxb) {

		/* horizontal window */
		ix = NINT((xb-fx)/dtr);
		ixlo = MAX(ix-NINT(xwh/dtr),0);
		ixhi = MIN(ix+NINT(xwh/dtr),ntr-1);
		nxw = 1+ixhi-ixlo;
		fxw = fx+(ixlo-ix)*dtr;

		rayxp=firp+NINT((xb-fx)/dx);  //ray initial point of each beam center//

/*  Ensure initial point of ray-cdp at the range of velocity model*/
		if((rayxp>1)&&(rayxp<nx))
		{

		fprintf(stderr,"ixb/nxb = %d/%d  ix = %d\n",ixb,nxb,ixlo);

		/* allocate space for beams */
		b = alloc2float(ntau,npx);

		zero2float(b,ntau,npx);
		// form beams at surface
		formBeams(bwh,dxb,fmin,
			nt,dt,ft,nxw,dtr,fxw,&f[ixlo],
			ntau,dtau,ftau,npx,dpx,fpx,b);

    	cell2=(Cell***)alloc3(mz,mx,npx,sizeof(Cell));
		for (ipx=0; ipx<npx; ++ipx)
            for(jx=0;jx<mx;jx++)
		        for(jz=0;jz<mz;jz++)
              {
				cell2[ipx][jx][jz].live=0;
				cell2[ipx][jx][jz].ip=0;
				cell2[ipx][jx][jz].tr=0.0;
				cell2[ipx][jx][jz].ti=0.0;
				cell2[ipx][jx][jz].ar=0.0;
				cell2[ipx][jx][jz].ai=0.0;
				cell2[ipx][jx][jz].angle=0.0;
			  }

			/* loop over beams */
		for (ipx=0,px=fpx+0*dpx; ipx<npx;++ipx,px+=dpx)
		{

			/* sine of emergence angle; skip if out of bounds */
			if (px*v[rayxp-1][0]>sin(amax)+0.01) continue;
			if (px*v[rayxp-1][0]<sin(amin)-0.01) continue;

			/* emergence angle and location */
			a0 = -asin(px*v[rayxp-1][0]);
			x0 = (rayxp-1)*dx;
			z0 = fz;
//	    	printf("ixlo=%f\n",x0);
			/* beam half-width adjusted for cosine of angle */
			bwhc = bwh*cos(a0);

			/* trace ray */

			ray2 = makeRay(x0,z0,a0,nt,dt,ft,nx,dx,fx,nz,dz,fz,v,del,ea,vpphase,vpgroup);
			/* accumulate ray to the coarse grid--cell */
		   accray(ray2,cell2[ipx],fmin,bwhc,lx,lz,mx,mz,live,ipx,ntau,dtau,ftau,
				nx,dx,fx,nz,dz,fz,g,v);
			/* free ray */
			freeRay(ray2);

		}

	    scanimg(cell1, cell2, live, dead, ntau, dtau, ftau,
   	fmin, lx, lz, nx, nz, mx, mz, npx, b, g, adcig, fpx, dpx);

          free2float(b);

	  free3((void***)cell2);
          }
	}
	free3((void***)cell1);
}


/* circle for efficiently finding nearest ray step */
typedef struct CircleStruct
{
	int irsf;               /* index of first raystep in circle */
	int irsl;               /* index of last raystep in circle */
	float x;                /* x coordinate of center of circle */
	float z;                /* z coordinate of center of circle */
	float r;                /* radius of circle */
} Circle;

/* functions defined and used internally */
Circle *makeCircles (int nc, int nrs, RayStep *rs);

Ray *makeRay (float x0, float z0, float a0, int nt, float dt, float ft,
	int nx, float dx, float fx, int nz, float dz, float fz, float **vxz,float **delxz,
	float **eaxz,float **vpphasexz,float **vpgroupxz)
/*****************************************************************************
Trace a ray for uniformly sampled v(x,z).
******************************************************************************
      nput:
x0		x coordinate of takeoff point
z0		z coordinate of takeoff point
a0		takeoff angle (radians)
nt		number of time samples
dt		time sampling interval
ft		first time sample
nx		number of x samples
dx		x sampling interval
fx		first x sample
nz		number of z samples
dz		z sampling interval
fz		first z sample
vxz		array[nx][nz] of uniformly sampled velocities v(x,z)
Pvpphase     P
P  at     angle
p     dvpdag
            Pdvpda  a=phaseangle
P(Vp)2=VpVpVpVp=vpphase*vpphase
 vpgroup
  apgroup p angle group

Returned:	pointer to ray parameters sampled at discrete ray steps
******************************************************************************
Notes:
The ray ends when it runs out of time (after nt steps) or with the first
step that is out of the (x,z) bounds of the velocity function v(x,z).
*****************************************************************************/
{
    /*if you are very interested in it, please contact me through email:
    18266200368@163.com*/
}

/******************************************************************************/
/* Input the data of shot gather and determine source and first receiver point*/
/******************************************************************************/
void inputrace(int is, int nt, float dx, int maxtr, FILE *fp, int *sisp, int *firisp, int *nistr)
{

	int itr,i,ntrace;
	int nshot;
	int *head;
	int offsetmin;
	int cdpmin;
	offsetmin=10000;
	head=alloc1int(60);
	for(i=0;i<60;i++)
		head[i]=0;
	rewind(fp);
	for(itr=0;itr<maxtr;itr++)
	{

	  fread(head,sizeof(int),60,fp);
	  fseek(fp,4*nt,1);
	  if(head[2]==is)
	  {
		  fseek(fp,-4*(nt+60),1);
		  break;
	  }
	}
	ntrace=0;
	for(i=0;i<60;i++)
		head[i]=0;
	for(i=itr;i<maxtr;i++)
	{
	  fread(head,sizeof(int),60,fp);
	  fseek(fp,4*nt,1);
	  if(abs(head[9])<abs(offsetmin))
	  {
		  offsetmin=head[9];
		  //printf("offsetmin=%d\n",offsetmin);
		  cdpmin=head[5];
		  //printf("cdpmin=%d\n",cdpmin);
	  }
	  if(head[2]==is)
		  ntrace++;
	  else
		  break;
	}
	*nistr=ntrace;
//  printf("%d\n",*nistr);

	if(offsetmin<0)
		*sisp=cdpmin+NINT(abs(offsetmin)/2/dx);
        else
		*sisp=cdpmin-NINT(abs(offsetmin)/2/dx);
//    printf("%d\n",*sisp);
	if(i==maxtr)
		fseek(fp,-4*(nt+60)*(*nistr),1);
	else
		fseek(fp,-4*(nt+60)*(*nistr+1),1);
	fread(head,sizeof(int),60,fp);
	*firisp=2*head[5]-*sisp;
	fseek(fp,-4*60,1);
	free1int(head);
}

void freeRay (Ray *ray)
/*****************************************************************************
Free a ray.
******************************************************************************
      nput:
ray		ray to be freed
*****************************************************************************/
{
	if (ray->c!=NULL) free1((void*)ray->c);
	free1((void*)ray->rs);
	free1((void*)ray);
}

int nearestRayStep (Ray *ray, float x, float z)
/*****************************************************************************
Determine index of ray step nearest to point (x,z).
******************************************************************************
      nput:
ray		ray
x		x coordinate
z		z coordinate

Returned:	index of nearest ray step
*****************************************************************************/
{
	int nrs=ray->nrs,ic=ray->ic,nc=ray->nc;
	RayStep *rs=ray->rs;
	Circle *c=(Circle *)ray->c;
	int irs,irsf,irsl,irsmin=0,update,jc,js,kc;
	float dsmin,ds,dx,dz,dmin,rdmin,xrs,zrs;

	/* if necessary, make circles localizing ray steps */
	if (c==NULL) {
		ray->ic = ic = 0;
		ray->nc = nc = sqrt((float)nrs);
		ray->c = c = makeCircles(nc,nrs,rs);
	}

	/* initialize minimum distance and minimum distance-squared */
	dx = x-c[ic].x;
	dz = z-c[ic].z;
	dmin = 2.0*(sqrt(dx*dx+dz*dz)+c[ic].r);
	dsmin = dmin*dmin;

	/* loop over all circles */
	for (kc=0,jc=ic,js=0; kc<nc; ++kc) {

		/* distance-squared to center of circle */
		dx = x-c[jc].x;
		dz = z-c[jc].z;
		ds = dx*dx+dz*dz;

		/* radius of circle plus minimum distance (so far) */
		rdmin = c[jc].r+dmin;

		/* if circle could possible contain a nearer ray step */
		if (ds<=rdmin*rdmin) {

			/* search circle for nearest ray step */
			irsf = c[jc].irsf;
			irsl = c[jc].irsl;
			update = 0;
			for (irs=irsf; irs<=irsl; ++irs) {
				xrs = rs[irs].x;
				zrs = rs[irs].z;
				dx = x-xrs;
				dz = z-zrs;
				ds = dx*dx+dz*dz;
				if (ds<dsmin) {
					dsmin = ds;
					irsmin = irs;
					update = 1;
				}
			}

			/* if a nearer ray step was found inside circle */
			if (update) {

				/* update minimum distance */
				dmin = sqrt(dsmin);

				/* remember the circle */
				ic = jc;
			}
		}

		/* search circles in alternating directions */
		js = (js>0)?-js-1:-js+1;
		jc += js;
		if (jc<0 || jc>=nc)
		{
			js = (js>0)?-js-1:-js+1;
			jc += js;
		}
	}

	/* remember the circle containing the nearest ray step */
	ray->ic = ic;

	if (irsmin<0 || irsmin>=nrs)
		fprintf(stderr,"irsmin=%d\n",irsmin);

	/* return index of nearest ray step */
	return irsmin;
}

int xxx_nearestRayStep (Ray *ray, float x, float z)
/*****************************************************************************
Determine index of ray step nearest to point (x,z).  Simple (slow) version.
******************************************************************************
      nput:
ray		ray
x		x coordinate
z		z coordinate

Returned:	index of nearest ray step
*****************************************************************************/
{
	int nrs=ray->nrs;
	RayStep *rs=ray->rs;
	int irs,irsmin=0;
	float dsmin,ds,dx,dz,xrs,zrs;

	for (irs=0,dsmin=FLT_MAX; irs<nrs; ++irs) {
		xrs = rs[irs].x;
		zrs = rs[irs].z;
		dx = x-xrs;
		dz = z-zrs;
		ds = dx*dx+dz*dz;
		if (ds<dsmin) {
			dsmin = ds;
			irsmin = irs;
		}
	}
	return irsmin;
}

Circle *makeCircles (int nc, int nrs, RayStep *rs)
/*****************************************************************************
Make circles used to speed up determination of nearest ray step.
******************************************************************************
      nput:
nc		number of circles to make
nrs		number of ray steps
rs		array[nrs] of ray steps

Returned:	array[nc] of circles
*****************************************************************************/
{
	int nrsc,ic,irsf,irsl,irs;
	float xmin,xmax,zmin,zmax,x,z,r;
	Circle *c;

	/* allocate space for circles */
	c = (Circle*)alloc1(nc,sizeof(Circle));

	/* determine typical number of ray steps per circle */
	nrsc = 1+(nrs-1)/nc;

	/* loop over circles */
	for (ic=0; ic<nc; ++ic) {

		/* index of first and last raystep */
		irsf = ic*nrsc;
		irsl = irsf+nrsc-1;
		if (irsf>=nrs) irsf = nrs-1;
		if (irsl>=nrs) irsl = nrs-1;

		/* coordinate bounds of ray steps */
		xmin = xmax = rs[irsf].x;
		zmin = zmax = rs[irsf].z;
		for (irs=irsf+1; irs<=irsl; ++irs) {
			if (rs[irs].x<xmin) xmin = rs[irs].x;
			if (rs[irs].x>xmax) xmax = rs[irs].x;
			if (rs[irs].z<zmin) zmin = rs[irs].z;
			if (rs[irs].z>zmax) zmax = rs[irs].z;
		}

		/* center and radius of circle */
		x = 0.5*(xmin+xmax);
		z = 0.5*(zmin+zmax);
		r = sqrt((x-xmin)*(x-xmin)+(z-zmin)*(z-zmin));

		/* set circle */
		c[ic].irsf = irsf;
		c[ic].irsl = irsl;
		c[ic].x = x;
		c[ic].z = z;
		c[ic].r = r;
	}

	return c;
}

/*****************************************************************************
Functions to support interpolation of velocity and its derivatives.
******************************************************************************
Functions:
vel2Alloc	allocate and initialize an interpolator for v(x,z)
vel2Interp	interpolate v(x,z) and its derivatives
******************************************************************************
Notes:
      interpolation is performed by piecewise cubic Hermite polynomials
so that velocity and first derivatives are continuous.  Therefore,
velocity v, first derivatives dv/dx and dv/dz, and the mixed
derivative ddv/dxdz are continuous.However, second derivatives
ddv/dxdx and ddv/dzdz are discontinuous.ddv/dxdx and ddv/dzdz
*****************************************************************************/



/* number of pre-computed, tabulated interpolators */
#define NTABLE 101

/* length of each interpolator in table (4 for piecewise cubic) */
#define LTABLE 4

/* table of pre-computed interpolators, for 0th, 1st, and 2nd derivatives */
static float tbl[3][NTABLE][LTABLE];

/* constants */
static int ix=1-LTABLE/2-LTABLE,iz=1-LTABLE/2-LTABLE;
static float ltable=LTABLE,ntblm1=NTABLE-1;

/* indices for 0th, 1st, and 2nd derivatives */
static int kx[6]={0,1,0,2,1,0};
static int kz[6]={0,0,1,0,1,2};
/* indices for 0th, 1st, and 2nd derivatives */
static int kkx[5]={1,2,0,0,1};
static int kkz[5]={0,0,1,2,1};
/* indices for 0th, 1st, and 2nd derivatives*/
static int kkkx[2]={1,0};
static int kkkz[2]={0,1};


/* function to build interpolator tables; sets tabled=1 when built */
static void buildTables (void);
static int tabled=0;

/* interpolator for velocity function v(x,z) of two variables */

typedef struct Vel2Struct
{
	int nx;		/* number of x samples */
	int nz;		/* number of z samples */
	int nxm;	/* number of x samples minus LTABLE */
	int nzm;	/* number of z samples minus LTABLE */
	float xs,xb,zs,zb,sx[3],sz[3],**vu;

    /* float **vupphase; */
    /* float **vupgroup; */
    float **duel,**eua;

} Vel2;


typedef struct Vel3Struct
{
	int nx;		/* number of x samples */
	int nz;		/* number of z samples */
	int nxm;	/* number of x samples minus LTABLE */
	int nzm;	/* number of z samples minus LTABLE */
	float xs,xb,zs,zb,sx[3],sz[3];

    float **vupphase;
    /*float **vupgroup; */
    /*float **duel,**eua; */
} Vel3;


typedef struct Vel4Struct
{
	int nx;		/* number of x samples */
	int nz;		/* number of z samples */
	int nxm;	/* number of x samples minus LTABLE */
	int nzm;	/* number of z samples minus LTABLE */
	float xs,xb,zs,zb,sx[3],sz[3];

    /*float **vupphase; */
    float **vupgroup;
    /*float **duel,**eua; */
} Vel4;

void* vel2Alloc (int nx, float dx, float fx,
	int nz, float dz, float fz, float **v,float **del,float **ea)
/*****************************************************************************
Allocate and initialize an interpolator for v(x,z) and its derivatives.
******************************************************************************
      nput:
nx		number of x samples
dx		x sampling interval
fx		first x sample
nz		number of z samples
dz		z sampling interval
fz		first z sample
v		array[nx][nz] of uniformly sampled v(x,z)
del
ea
Returned:	pointer to interpolator
*****************************************************************************/
{
	Vel2 *vel2;

	/* allocate space */
	vel2 = (Vel2*)alloc1(1,sizeof(Vel2));

	/* set state variables used for interpolation */
	vel2->nx = nx;
	vel2->nxm = nx-LTABLE;
	vel2->xs = 1.0/dx;
	vel2->xb = ltable-fx*vel2->xs;
	/*1.0,1/dx,1/(dx*dx)*/
	vel2->sx[0] = 1.0;
	vel2->sx[1] = vel2->xs;
	vel2->sx[2] = vel2->xs*vel2->xs;
	vel2->nz = nz;
	vel2->nzm = nz-LTABLE;
	vel2->zs = 1.0/dz;
	vel2->zb = ltable-fz*vel2->zs;
	/*1.0,1/dz,1/(dz*dz)*/
	vel2->sz[0] = 1.0;
	vel2->sz[1] = vel2->zs;
	vel2->sz[2] = vel2->zs*vel2->zs;
	/*vvu*/
	vel2->vu = v;
    /*del\A3\ACeaduel,eua*/
    vel2->duel = del;
	vel2->eua = ea;
	/* if necessary, build interpolator coefficient tables */
	if (!tabled) buildTables();
	return vel2;
}

void* vel3Alloc (int nx, float dx, float fx,
	int nz, float dz, float fz, float **vpphase)
/*****************************************************************************
Allocate and initialize an interpolator for v(x,z) and its derivatives.
******************************************************************************
      nput:
nx		number of x samples
dx		x sampling interval
fx		first x sample
nz		number of z samples
dz		z sampling interval
fz		first z sample
vpphase		array[nx][nz] of uniformly sampled v(x,z)

Returned:	pointer to interpolator
*****************************************************************************/
{
	Vel3 *vel3;

	/* allocate space */
	vel3 = (Vel3*)alloc1(1,sizeof(Vel3));

	/* set state variables used for interpolation*/
	vel3->nx = nx;
	vel3->nxm = nx-LTABLE;
	vel3->xs = 1.0/dx;
	vel3->xb = ltable-fx*vel3->xs;
	/*1.0,1/dx,1/(dx*dx)*/
	vel3->sx[0] = 1.0;
	vel3->sx[1] = vel3->xs;
	vel3->sx[2] = vel3->xs*vel3->xs;
	vel3->nz = nz;
	vel3->nzm = nz-LTABLE;
	vel3->zs = 1.0/dz;
	vel3->zb = ltable-fz*vel3->zs;
	/*1.0,1/dz,1/(dz*dz)*/
	vel3->sz[0] = 1.0;
	vel3->sz[1] = vel3->zs;
	vel3->sz[2] = vel3->zs*vel3->zs;
	/*vvu*/
	vel3->vupphase = vpphase;

	/* if necessary, build interpolator coefficient tables */
	if (!tabled) buildTables();
	return vel3;
}


void* vel4Alloc (int nx, float dx, float fx,
	int nz, float dz, float fz, float **vpgroup)
/*****************************************************************************
Allocate and initialize an interpolator for v(x,z) and its derivatives.
******************************************************************************
      nput:
nx		number of x samples
dx		x sampling interval
fx		first x sample
nz		number of z samples
dz		z sampling interval
fz		first z sample
vpgroup		array[nx][nz] of uniformly sampled v(x,z)

Returned:	pointer to interpolator
*****************************************************************************/
{
	Vel4 *vel4;

	/* allocate space */
	vel4 = (Vel4*)alloc1(1,sizeof(Vel4));

	/* set state variables used for interpolation */
	vel4->nx = nx;
	vel4->nxm = nx-LTABLE;
	vel4->xs = 1.0/dx;
	vel4->xb = ltable-fx*vel4->xs;
	/*1.0,1/dx,1/(dx*dx)*/
	vel4->sx[0] = 1.0;
	vel4->sx[1] = vel4->xs;
	vel4->sx[2] = vel4->xs*vel4->xs;
	vel4->nz = nz;
	vel4->nzm = nz-LTABLE;
	vel4->zs = 1.0/dz;
	vel4->zb = ltable-fz*vel4->zs;
	/*1.0,1/dz,1/(dz*dz)*/
	vel4->sz[0] = 1.0;
	vel4->sz[1] = vel4->zs;
	vel4->sz[2] = vel4->zs*vel4->zs;
	/*vpgroup-vupgroup*/
	vel4->vupgroup = vpgroup;

	/* if necessary, build interpolator coefficient tables */
	if (!tabled) buildTables();
	return vel4;
}


void vel2Free (void *vel2)
/*****************************************************************************
Free an interpolator for v(x,z) and its derivatives.
******************************************************************************
      nput:
vel2		pointer to interpolator as returned by vel2Alloc()
*****************************************************************************/
{
	free1(vel2);
}
void vel3Free (void *vel3)
/*****************************************************************************
Free an interpolator for v(x,z) and its derivatives.
******************************************************************************
      nput:
vel2		pointer to interpolator as returned by vel2Alloc()
*****************************************************************************/
{
	free1(vel3);
}
void vel4Free (void *vel4)
/*****************************************************************************
Free an interpolator for v(x,z) and its derivatives.
******************************************************************************
      nput:
vel2		pointer to interpolator as returned by vel2Alloc()
*****************************************************************************/
{
	free1(vel4);
}

void vel3Interp (void *vel3, float x, float z, float *vpphasex, float *vpphasez, float *vpphasexx, float *vpphasezz,float *vpphasexz)
/*****************************************************************************
      interpolation of a velocity function v(x,z) and its derivatives.

******************************************************************************
      nput:
vel3		pointer to interpolator as returned by vel2Alloc()
x		x coordinate at which to interpolate v(x,z) and derivatives
z		z coordinate at which to interpolate v(x,z) and derivatives

Output:
vpphasex		dvpphase/dx
vpphasez		dvpphase/dz
vpphasexx		ddvpphase/dxdx
vpphasezz		ddvpphase/dzdz
vpphasexz       ddvpphase/dxdz
*****************************************************************************/
{
	Vel3 *v3=(Vel3 *)vel3;
	int nx=v3->nx,nz=v3->nz,nxm=v3->nxm,nzm=v3->nzm;
	float xs=v3->xs,xb=v3->xb,zs=v3->zs,zb=v3->zb,
		*sx=v3->sx,*sz=v3->sz,**vupphase=v3->vupphase;
	int i,jx,lx,mx,jz,lz,mz,jmx,jmz,mmx,mmz;
	float ax,bx,*px,az,bz,*pz,sum,vpphased[5];
	/*xbΪ\CA\F8\D6\D0\D0ĵ\C4λ\D6ã\ACxs=1/dx,ax??*/

	/* determine offsets into vu and interpolation coefficients */
	ax = xb+x*xs;
	jx = (int)ax;
	bx = ax-jx;
	lx = (bx>=0.0)?bx*ntblm1+0.5:(bx+1.0)*ntblm1-0.5;

	lx *= LTABLE;
	mx = ix+jx;
	az = zb+z*zs;
	jz = (int)az;
	bz = az-jz;
	lz = (bz>=0.0)?bz*ntblm1+0.5:(bz+1.0)*ntblm1-0.5;
	lz *= LTABLE;
	mz = iz+jz;

	/*int*/
//	printf("lx=%d , lz=%d \n",lx,lz);

	/* if totally within input array, use fast method */
	if (mx>=0 && mx<=nxm && mz>=0 && mz<=nzm)
	{
		for (i=0; i<5; ++i)
		{
			px = &(tbl[kkx[i]][0][0])+lx;
			pz = &(tbl[kkz[i]][0][0])+lz;

			/*float*/
	      // printf("px=%f , pz=%f \n",*px,*pz);

			vpphased[i] = sx[kkx[i]]*sz[kkz[i]]*(
				vupphase[mx][mz]*px[0]*pz[0]+
				vupphase[mx][mz+1]*px[0]*pz[1]+
				vupphase[mx][mz+2]*px[0]*pz[2]+
				vupphase[mx][mz+3]*px[0]*pz[3]+
				vupphase[mx+1][mz]*px[1]*pz[0]+
				vupphase[mx+1][mz+1]*px[1]*pz[1]+
				vupphase[mx+1][mz+2]*px[1]*pz[2]+
				vupphase[mx+1][mz+3]*px[1]*pz[3]+
				vupphase[mx+2][mz]*px[2]*pz[0]+
				vupphase[mx+2][mz+1]*px[2]*pz[1]+
				vupphase[mx+2][mz+2]*px[2]*pz[2]+
				vupphase[mx+2][mz+3]*px[2]*pz[3]+
				vupphase[mx+3][mz]*px[3]*pz[0]+
				vupphase[mx+3][mz+1]*px[3]*pz[1]+
				vupphase[mx+3][mz+2]*px[3]*pz[2]+
				vupphase[mx+3][mz+3]*px[3]*pz[3]);
		}

	/* else handle end effects with constant extrapolation */
	}
	else
	{
		for (i=0; i<5; ++i)
		{
			px = &(tbl[kkx[i]][0][0])+lx;
			pz = &(tbl[kkz[i]][0][0])+lz;

	       /*float*/
	      // printf("px=%f , pz=%f \n",px,pz);

			for (jx=0,jmx=mx,sum=0.0; jx<4; ++jx,++jmx)
			{
				mmx = jmx;
				if (mmx<0) mmx = 0;
				else if (mmx>=nx) mmx = nx-1;
				for (jz=0,jmz=mz; jz<4; ++jz,++jmz)
				{
					mmz = jmz;
					if (mmz<0) mmz = 0;
					else if (mmz>=nz) mmz = nz-1;
					sum += vupphase[mmx][mmz]*px[jx]*pz[jz];
				}
			}
			vpphased[i] = sx[kkx[i]]*sz[kkz[i]]*sum;
		}
	}

	/* set output variables */
	*vpphasex = vpphased[0];
	*vpphasez = vpphased[1];
	*vpphasexx = vpphased[2];
	*vpphasezz = vpphased[3];
	*vpphasexz = vpphased[4];
}
void vel2Interp (void *vel2, float x, float z,
	float *v, float *vx, float *vz, float *vxx, float *vxz, float *vzz,float *del,float *ea)
/*****************************************************************************
      interpolation of a velocity function v(x,z) and its derivatives.

******************************************************************************
      nput:
vel2		pointer to interpolator as returned by vel2Alloc()vel2Alloc
x		x coordinate at which to interpolate v(x,z) and derivatives
z		z coordinate at which to interpolate v(x,z) and derivatives

Output:
v		v(x,z)
vx		dv/dx
vz		dv/dz
vxx		ddv/dxdx
vxz		ddv/dxdz
vzz		ddv/dzdz
del     del(x,z)
ea      ea(x,z)
*****************************************************************************/
{
	Vel2 *v2=(Vel2 *)vel2;
	int nx=v2->nx,nz=v2->nz,nxm=v2->nxm,nzm=v2->nzm;
	float xs=v2->xs,xb=v2->xb,zs=v2->zs,zb=v2->zb,
		*sx=v2->sx,*sz=v2->sz,**vu=v2->vu;
	float **duel=v2->duel,**eua=v2->eua;
	int i,jx,lx,mx,jz,lz,mz,jmx,jmz,mmx,mmz;
	float ax,bx,*px,az,bz,*pz,sum,vd[6];
	float dueld, euad;


	/* determine offsets into vu and interpolation coefficients */

	/*xbxs=1/dx,ax??*/
 //   printf("output xb=%f \n",xb);

	ax = xb+x*xs;

 //   printf("output ax=%f \n",ax);

	jx = (int)ax;

//	printf("output jx=%d \n",jx);

	bx = ax-jx;

//	printf("output bx=%f \n",bx);


	lx = (bx>=0.0)?bx*ntblm1+0.5:(bx+1.0)*ntblm1-0.5;

   // printf("output lx=%f \n",lx);

	lx *= LTABLE;
	mx = ix+jx;
	az = zb+z*zs;
	jz = (int)az;
	bz = az-jz;
	lz = (bz>=0.0)?bz*ntblm1+0.5:(bz+1.0)*ntblm1-0.5;
	lz *= LTABLE;
	mz = iz+jz;


//	printf("lx=%d , lz=%d \n",lx,lz);



	if (mx>=0 && mx<=nxm && mz>=0 && mz<=nzm)
	{
		for (i=0; i<6; ++i)
		{
			px = &(tbl[kx[i]][0][0])+lx;
			pz = &(tbl[kz[i]][0][0])+lz;


	      // printf("px=%f , pz=%f \n",*px,*pz);

			vd[i] = sx[kx[i]]*sz[kz[i]]*(
				vu[mx][mz]*px[0]*pz[0]+
				vu[mx][mz+1]*px[0]*pz[1]+
				vu[mx][mz+2]*px[0]*pz[2]+
				vu[mx][mz+3]*px[0]*pz[3]+
				vu[mx+1][mz]*px[1]*pz[0]+
				vu[mx+1][mz+1]*px[1]*pz[1]+
				vu[mx+1][mz+2]*px[1]*pz[2]+
				vu[mx+1][mz+3]*px[1]*pz[3]+
				vu[mx+2][mz]*px[2]*pz[0]+
				vu[mx+2][mz+1]*px[2]*pz[1]+
				vu[mx+2][mz+2]*px[2]*pz[2]+
				vu[mx+2][mz+3]*px[2]*pz[3]+
				vu[mx+3][mz]*px[3]*pz[0]+
				vu[mx+3][mz+1]*px[3]*pz[1]+
				vu[mx+3][mz+2]*px[3]*pz[2]+
				vu[mx+3][mz+3]*px[3]*pz[3]);
		}

	/* else handle end effects with constant extrapolation */
	}
	else
	{
		for (i=0; i<6; ++i)
		{
			px = &(tbl[kx[i]][0][0])+lx;
			pz = &(tbl[kz[i]][0][0])+lz;
			for (jx=0,jmx=mx,sum=0.0; jx<4; ++jx,++jmx)
			{
				mmx = jmx;
				if (mmx<0) mmx = 0;
				else if (mmx>=nx) mmx = nx-1;
				for (jz=0,jmz=mz; jz<4; ++jz,++jmz)
				{
					mmz = jmz;
					if (mmz<0) mmz = 0;
					else if (mmz>=nz) mmz = nz-1;
					sum += vu[mmx][mmz]*px[jx]*pz[jz];
				}
			}
			vd[i] = sx[kx[i]]*sz[kz[i]]*sum;
		}
	}

	/* if totally within input array, use fast method,float dueld, euad;*/
	if (mx>=0 && mx<=nxm && mz>=0 && mz<=nzm)
	{

			px = &(tbl[kx[0]][0][0])+lx;
			pz = &(tbl[kz[0]][0][0])+lz;

			dueld = sx[kx[0]]*sz[kz[0]]*(
				duel[mx][mz]*px[0]*pz[0]+
				duel[mx][mz+1]*px[0]*pz[1]+
				duel[mx][mz+2]*px[0]*pz[2]+
				duel[mx][mz+3]*px[0]*pz[3]+
				duel[mx+1][mz]*px[1]*pz[0]+
				duel[mx+1][mz+1]*px[1]*pz[1]+
				duel[mx+1][mz+2]*px[1]*pz[2]+
				duel[mx+1][mz+3]*px[1]*pz[3]+
				duel[mx+2][mz]*px[2]*pz[0]+
				duel[mx+2][mz+1]*px[2]*pz[1]+
				duel[mx+2][mz+2]*px[2]*pz[2]+
				duel[mx+2][mz+3]*px[2]*pz[3]+
				duel[mx+3][mz]*px[3]*pz[0]+
				duel[mx+3][mz+1]*px[3]*pz[1]+
				duel[mx+3][mz+2]*px[3]*pz[2]+
				duel[mx+3][mz+3]*px[3]*pz[3]);


	/* else handle end effects with constant extrapolation */
	}
	else
	{
			px = &(tbl[kx[0]][0][0])+lx;
			pz = &(tbl[kz[0]][0][0])+lz;
			for (jx=0,jmx=mx,sum=0.0; jx<4; ++jx,++jmx)
			{
				mmx = jmx;
				if (mmx<0) mmx = 0;
				else if (mmx>=nx) mmx = nx-1;
				for (jz=0,jmz=mz; jz<4; ++jz,++jmz)
				{
					mmz = jmz;
					if (mmz<0) mmz = 0;
					else if (mmz>=nz) mmz = nz-1;
					sum += duel[mmx][mmz]*px[jx]*pz[jz];
				}
			}
			dueld = sx[kx[i]]*sz[kz[i]]*sum;

	}

	if (mx>=0 && mx<=nxm && mz>=0 && mz<=nzm)
	{

			px = &(tbl[kx[0]][0][0])+lx;
			pz = &(tbl[kz[0]][0][0])+lz;

				euad = sx[kx[0]]*sz[kz[0]]*(
				eua[mx][mz]*px[0]*pz[0]+
				eua[mx][mz+1]*px[0]*pz[1]+
				eua[mx][mz+2]*px[0]*pz[2]+
				eua[mx][mz+3]*px[0]*pz[3]+
				eua[mx+1][mz]*px[1]*pz[0]+
				eua[mx+1][mz+1]*px[1]*pz[1]+
				eua[mx+1][mz+2]*px[1]*pz[2]+
				eua[mx+1][mz+3]*px[1]*pz[3]+
				eua[mx+2][mz]*px[2]*pz[0]+
				eua[mx+2][mz+1]*px[2]*pz[1]+
				eua[mx+2][mz+2]*px[2]*pz[2]+
				eua[mx+2][mz+3]*px[2]*pz[3]+
				eua[mx+3][mz]*px[3]*pz[0]+
				eua[mx+3][mz+1]*px[3]*pz[1]+
				eua[mx+3][mz+2]*px[3]*pz[2]+
				eua[mx+3][mz+3]*px[3]*pz[3]);


	/* else handle end effects with constant extrapolation */
	}
	else
	{
			px = &(tbl[kx[0]][0][0])+lx;
			pz = &(tbl[kz[0]][0][0])+lz;
			for (jx=0,jmx=mx,sum=0.0; jx<4; ++jx,++jmx)
			{
				mmx = jmx;
				if (mmx<0) mmx = 0;
				else if (mmx>=nx) mmx = nx-1;
				for (jz=0,jmz=mz; jz<4; ++jz,++jmz)
				{
					mmz = jmz;
					if (mmz<0) mmz = 0;
					else if (mmz>=nz) mmz = nz-1;
					sum += eua[mmx][mmz]*px[jx]*pz[jz];
				}
			}
			euad = sx[kx[i]]*sz[kz[i]]*sum;

	}
	/* set output variables */
	*v = vd[0];
	*vx = vd[1];
	*vz = vd[2];
	*vxx = vd[3];
	*vxz = vd[4];
	*vzz = vd[5];
    *del=dueld;
	*ea=euad;
}

void vel4Interp (void *vel4, float x, float z, float *vpgroupx, float *vpgroupz)
/*****************************************************************************
      interpolation of a velocity function v(x,z) and its derivatives.

******************************************************************************
      nput:
vel4		pointer to interpolator as returned by vel2Alloc()
x		x coordinate at which to interpolate v(x,z) and derivatives
z		z coordinate at which to interpolate v(x,z) and derivatives

Output:
vpgroupx		dvpgroup/dx
vpgroupz		dvpgroup/dz
*****************************************************************************/
{
	Vel4 *v4=(Vel4 *)vel4;
	int nx=v4->nx,nz=v4->nz,nxm=v4->nxm,nzm=v4->nzm;
	float xs=v4->xs,xb=v4->xb,zs=v4->zs,zb=v4->zb,
		*sx=v4->sx,*sz=v4->sz,**vupgroup=v4->vupgroup;
	int i,jx,lx,mx,jz,lz,mz,jmx,jmz,mmx,mmz;
	float ax,bx,*px,az,bz,*pz,sum,vupgroupd[2];


	/* determine offsets into vu and interpolation coefficients */
	ax = xb+x*xs;
	jx = (int)ax;
	bx = ax-jx;
	lx = (bx>=0.0)?bx*ntblm1+0.5:(bx+1.0)*ntblm1-0.5;

	lx *= LTABLE;
	mx = ix+jx;
	az = zb+z*zs;
	jz = (int)az;
	bz = az-jz;
	lz = (bz>=0.0)?bz*ntblm1+0.5:(bz+1.0)*ntblm1-0.5;
	lz *= LTABLE;
	mz = iz+jz;


//	printf("lx=%d , lz=%d \n",lx,lz);

	/* if totally within input array, use fast method */
	if (mx>=0 && mx<=nxm && mz>=0 && mz<=nzm)
	{
		for (i=0; i<2; ++i)
		{
			px = &(tbl[kkkx[i]][0][0])+lx;
			pz = &(tbl[kkkz[i]][0][0])+lz;


	      // printf("px=%f , pz=%f \n",*px,*pz);

			vupgroupd[i] = sx[kkx[i]]*sz[kkz[i]]*(
				vupgroup[mx][mz]*px[0]*pz[0]+
				vupgroup[mx][mz+1]*px[0]*pz[1]+
				vupgroup[mx][mz+2]*px[0]*pz[2]+
				vupgroup[mx][mz+3]*px[0]*pz[3]+
				vupgroup[mx+1][mz]*px[1]*pz[0]+
				vupgroup[mx+1][mz+1]*px[1]*pz[1]+
				vupgroup[mx+1][mz+2]*px[1]*pz[2]+
				vupgroup[mx+1][mz+3]*px[1]*pz[3]+
				vupgroup[mx+2][mz]*px[2]*pz[0]+
				vupgroup[mx+2][mz+1]*px[2]*pz[1]+
				vupgroup[mx+2][mz+2]*px[2]*pz[2]+
				vupgroup[mx+2][mz+3]*px[2]*pz[3]+
				vupgroup[mx+3][mz]*px[3]*pz[0]+
				vupgroup[mx+3][mz+1]*px[3]*pz[1]+
				vupgroup[mx+3][mz+2]*px[3]*pz[2]+
				vupgroup[mx+3][mz+3]*px[3]*pz[3]);
		}

	/* else handle end effects with constant extrapolation */
	}
	else
	{
		for (i=0; i<2; ++i)
		{
			px = &(tbl[kkkx[i]][0][0])+lx;
			pz = &(tbl[kkkz[i]][0][0])+lz;


	      // printf("px=%f , pz=%f \n",px,pz);

			for (jx=0,jmx=mx,sum=0.0; jx<4; ++jx,++jmx)
			{
				mmx = jmx;
				if (mmx<0) mmx = 0;
				else if (mmx>=nx) mmx = nx-1;
				for (jz=0,jmz=mz; jz<4; ++jz,++jmz)
				{
					mmz = jmz;
					if (mmz<0) mmz = 0;
					else if (mmz>=nz) mmz = nz-1;
					sum += vupgroup[mmx][mmz]*px[jx]*pz[jz];
				}
			}
			vupgroupd[i] = sx[kkx[i]]*sz[kkz[i]]*sum;
		}
	}

	/* set output variables */
	*vpgroupx = vupgroupd[0];
	*vpgroupz = vupgroupd[1];

}

/* hermite polynomials  hermite*/
static float h00 (float x) {return 2.0*x*x*x-3.0*x*x+1.0;}
static float h01 (float x) {return 6.0*x*x-6.0*x;}
static float h02 (float x) {return 12.0*x-6.0;}
static float h10 (float x) {return -2.0*x*x*x+3.0*x*x;}
static float h11 (float x) {return -6.0*x*x+6.0*x;}
static float h12 (float x) {return -12.0*x+6.0;}
static float k00 (float x) {return x*x*x-2.0*x*x+x;}
static float k01 (float x) {return 3.0*x*x-4.0*x+1.0;}
static float k02 (float x) {return 6.0*x-4.0;}
static float k10 (float x) {return x*x*x-x*x;}
static float k11 (float x) {return 3.0*x*x-2.0*x;}
static float k12 (float x) {return 6.0*x-2.0;}

/* function to build interpolation tables */
static void buildTables(void)
{
	int i;
	float x;

	/* tabulate interpolator for 0th derivative */
	for (i=0; i<NTABLE; ++i) {
		x = (float)i/(NTABLE-1.0);
		tbl[0][i][0] = -0.5*k00(x);
		tbl[0][i][1] = h00(x)-0.5*k10(x);
		tbl[0][i][2] = h10(x)+0.5*k00(x);
		tbl[0][i][3] = 0.5*k10(x);
		tbl[1][i][0] = -0.5*k01(x);
		tbl[1][i][1] = h01(x)-0.5*k11(x);
		tbl[1][i][2] = h11(x)+0.5*k01(x);
		tbl[1][i][3] = 0.5*k11(x);
		tbl[2][i][0] = -0.5*k02(x);
		tbl[2][i][1] = h02(x)-0.5*k12(x);
		tbl[2][i][2] = h12(x)+0.5*k02(x);
		tbl[2][i][3] = 0.5*k12(x);
	}

	/* remember that tables have been built */
	tabled = 1;
}

/* Beam subroutines */



/* functions defined and used internally */
static void xtop (float w,
	int nx, float dx, float fx, complex *g,
	int np, float dp, float fp, complex *h);
static BeamData* beamData (int npx, float wmin, int nt, float dt, float ft, float **f);
static void setCell (Cells *cells, int jx, int jz);
static void accCell (Cells *cells, int jx, int jz);
static int cellTimeAmp (Cells *cells, int jx, int jz);
static void getangle(Cell **c, int mx, int mz, float dx, float dz, int live);
static void cellBeam (Cell **cell, Cell ***c1, Cell ***c2, float **g, int jx, int jz, BeamData *bd, int lx, int lz,
					  int nx, int nz, int npx, int ips, int ipr, float ***adcig,float fpx,float dpx);



/* functions for external use */

void formBeams (float bwh, float dxb, float fmin,
	int nt, float dt, float ft,
	int nx, float dx, float fx, float **f,
	int ntau, float dtau, float ftau,
	int npx, float dpx, float fpx, float **g)
/*****************************************************************************
Form beams (filtered slant stacks) for later superposition of Gaussian beams.
******************************************************************************
      input:
bwh		horizontal beam half-width
dxb		horizontal distance between beam centers
fmin		minimum frequency (cycles per unit time)
nt		number of input time samples
dt		input time sampling interval
ft		first input time sample
nx		number of horizontal samples
dx		horizontal sampling interval
fx		first horizontal sample
f		array[nx][nt] of data to be slant stacked into beams
ntau		number of output time samples
dtau		output time sampling interval (currently must equal dt)
ftau		first output time sample
npx		number of horizontal slownesses
dpx		horizontal slowness sampling interval
fpx		first horizontal slowness

Output:
g		array[npx][ntau] containing beams
*****************************************************************************/
{
	int ntpad,ntfft,nw,ix,iw,ipx,it,itau;
	float wmin,pxmax,xmax,x,dw,fw,w,fftscl,
		amp,phase,scale,a,b,as,bs,es,cfr,cfi,
		*fpad,*gpad;
	complex **cf,**cg,*cfx,*cgpx;
	int nf1,nf2,nf3,nf4;
	float ffw,tmpp;

	/* minimum frequency in radians */
	wmin = 2.0*PI*fmin;

	/* pad time axis to avoid wraparound */
	pxmax = (dpx<0.0)?fpx:fpx+(npx-1)*dpx;
	xmax = (dx<0.0)?fx:fx+(nx-1)*dx;
	ntpad = ABS(pxmax*xmax)/dt;


	/* fft sampling */
	ntfft = npfar(MAX(nt+ntpad,ntau));
	nw = ntfft/2+1;
	dw = 2.0*PI/(ntfft*dt);
	fw = 0.0;
	fftscl = 1.0/ntfft;

	/* filter parameters*/
	ffw=2.0*PI*5;
	nf1=ffw/dw+0.5;
	ffw=2.0*PI*10;
	nf2=ffw/dw+0.5;
	ffw=2.0*PI*35;
	nf3=ffw/dw+0.5;
	ffw=2.0*PI*40;
	nf4=ffw/dw+0.5;

	/* allocate workspace */
	fpad = alloc1float(ntfft);
	gpad = alloc1float(ntfft);
	cf = alloc2complex(nw,nx);
	cg = alloc2complex(nw,npx);
	cfx = alloc1complex(nx);
	cgpx = alloc1complex(npx);

	zero1float(fpad,ntfft);
    zero1float(gpad,ntfft);
    zero2complex(cf,nw,nx);
    zero2complex(cg,nw,npx);
    zero1complex(cfx,nx);
    zero1complex(cgpx,npx);
	/* loop over x */
	for (ix=0; ix<nx; ++ix) {

		/* pad time with zeros */
		for (it=0; it<nt; ++it)
			fpad[it] = f[ix][it];
		for (it=nt; it<ntfft; ++it)
			fpad[it] = 0.0;

		/* Fourier transform time to frequency */
		pfarc(1,ntfft,fpad,cf[ix]);
	}


/*	for (ix=0; ix<nx; ++ix) {
 	     for(iw=0;iw<nw;iw++) {

		  if(iw<nf1||iw>nf4)
		    	cf[ix][iw]=cmplx(0.0,0.0);
		  else if(iw>=nf1&&iw<=nf2)
			    tmpp=0.54+0.46*cos(PI*(iw-nf1)/(nf2-nf1)-PI),
			    cf[ix][iw].r=cf[ix][iw].r*tmpp,
			    cf[ix][iw].i=cf[ix][iw].i*tmpp;

		   else if(iw>=nf3&&iw<=nf4)
                tmpp=0.54+0.46*cos(PI*(nf3-iw)/(nf4-nf3)),
		    	cf[ix][iw].r=cf[ix][iw].r*tmpp,
	     		cf[ix][iw].i=cf[ix][iw].i*tmpp;
	       else
		    	cf[ix][iw]=cf[ix][iw];
	   }
	}*/

	/* loop over w */
	for (iw=0,w=fw; iw<nw; ++iw,w+=dw) {

		/* frequency-dependent amplitude scale factors */
		scale = -0.5*w/(wmin*bwh*bwh);
		amp = fftscl*dpx*w/(2.0*PI)*sqrt(w/(PI*wmin))*dxb/bwh;

		/* phase shift to account for ft not equal to ftau */
		phase = w*(ft-ftau);

		/* apply complex filter */
		a = amp*cos(phase);
		b = amp*sin(phase);
		for (ix=0,x=fx; ix<nx; ++ix,x+=dx) {
			es = exp(scale*x*x);
			as = a*es;
			bs = b*es;
			cfr = cf[ix][iw].r;
			cfi = cf[ix][iw].i;
			cfx[ix].r = sqrt(2*w)*((as*cfr-bs*cfi)-(bs*cfr+as*cfi))/2;
			cfx[ix].i = sqrt(2*w)*((bs*cfr+as*cfi)+(as*cfr-bs*cfi))/2;
//			cfx[ix].r = -(bs*cfr+as*cfi)*w;
//			cfx[ix].i = (as*cfr-bs*cfi)*w;
		}

		/* transform x to p */
		xtop(w,nx,dx,fx,cfx,npx,dpx,fpx,cgpx);
		for (ipx=0; ipx<npx; ++ipx) {
			cg[ipx][iw].r = cgpx[ipx].r;
			cg[ipx][iw].i = cgpx[ipx].i;
		}
	}
	/* loop over px */
	for (ipx=0; ipx<npx; ++ipx) {

		/* Fourier transform frequency to time */
		pfacr(-1,ntfft,cg[ipx],gpad);

		/* copy to output array */
		for (itau=0; itau<ntau; ++itau)
			g[ipx][itau] = gpad[itau];
	}

	free1float(fpad);
	free1float(gpad);
	free2complex(cf);
	free2complex(cg);
	free1complex(cfx);
	free1complex(cgpx);
}

void accray (Ray *ray,Cell **c, float fmin, float lmin, int lx, int lz, int mx,
	int mz,int live, int ipx, int nt, float dt, float ft,
	int nx, float dx, float fx, int nz, float dz, float fz, float **g, float **v)
/*****************************************************************************
Accumulate contribution of one Gaussian beam.
******************************************************************************
      nput:
ray		ray parameters sampled at discrete ray steps
fmin		minimum frequency (cycles per unit time)
lmin		initial beam width for frequency wmin
nt		number of time samples
dt		time sampling interval
ft		first time sample
f		array[nt] containing data for one ray f(t)
nx		number of x samples
dx		x sampling interval
fx		first x sample
nz		number of z samples
dz		z sampling interval
fz		first z sample
g		array[nx][nz] in which to accumulate beam

Output:
g		array[nx][nz] after accumulating beam
*****************************************************************************/
{
	int jx,jz;
	int i;
	float wmin;
	RayStep *rs=ray->rs;
	Cells *cells;
	cells=(Cells*)alloc1(1,sizeof(Cells));
	/* frequency in radians per unit time */
	wmin = 2.0*PI*fmin;

	/* set information needed to set and fill cells */
	cells->nt = nt;
	cells->dt = dt;
	cells->ft = ft;
	cells->lx = lx;
	cells->mx = mx;
	cells->nx = nx;
	cells->dx = dx;
	cells->fx = fx;
	cells->lz = lz;
	cells->mz = mz;
	cells->nz = nz;
	cells->dz = dz;
	cells->fz = fz;
	cells->live = live;
	cells->ip = ipx;
	cells->wmin = wmin;
	cells->lmin = lmin;
	cells->cell = c;
	cells->ray = ray;
//	cells->bd = bd;
//	cells->g = g;


	/* cell closest to initial point on ray will be first live cell */
	jx = NINT((rs[0].x-fx)/dx/lx);
	jz = NINT((rs[0].z-fz)/dz/lz);

	/* set first live cell and its neighbors recursively */
	setCell(cells,jx,jz);
/*	getangle(c, mx, mz, dx, dz, live);
/*	for(jx=0;jx<mx-1;jx++)
		for(jz=0;jz<mz-1;jz++)
			if(c[jx][jz].live==live)
				printf("%f\n",c[jx][jz].angle);*/


	/* free complex beam data */
//	free2complex(bd->cf);
//	free1((void*)bd);

	/* free cells */
//	free2((void**)cells->cell);
	free1((void*)cells);
}

/* functions for internal use only */

static void xtop (float w,
	int nx, float dx, float fx, complex *g,
	int np, float dp, float fp, complex *h)
/*****************************************************************************
Slant stack for one frequency w, where slant stack is defined by

           fx+(nx-1)*dx
    h(p) =   integral   exp(-sqrt(-1)*w*p*x) * g(x) * dx
                fx
******************************************************************************
      nput:
w		frequency (radians per unit time)
nx		number of x samples
dx		x sampling interval
fx		first x sample
g		array[nx] containing g(x)
np		number of p samples
dp		p sampling interval
fp		first p sample

Output:
h		array[np] containing h(p)
******************************************************************************
Notes:
The units of w, x, and p must be consistent.

Slant stack is performed via FFT and 8-point (tapered-sinc) interpolation.

The Fourier transform over time (t) is assumed to have been performed with
positive sqrt(-1)*w*t in the exponent;  if negative sqrt(-1)*w*t was used
instead, call this function with negative w.
*****************************************************************************/
{
	int nxfft,nk,nka,ix,ik,ip,lwrap;
	float dk,fk,ek,fka,k,p,phase,c,s,x,xshift,temp,*kp;
	complex czero,*gx,*gk,*gka,*hp;
	czero=cmplx(0.0,0.0);

	/* number of samples required to make wavenumber k periodic */
	lwrap = 8;

	/* wavenumber k sampling */
	nxfft = npfa((nx+lwrap)*2);
	nk = nxfft;
	dk = 2.0*PI/(nxfft*dx);
	fk = -PI/dx;
	ek = PI/dx;
	fka = fk-lwrap*dk;
	nka = lwrap+nk+lwrap;

	/* allocate workspace */
	gka = alloc1complex(nka);
	gx = gk = gka+lwrap;
	hp = alloc1complex(np);
	kp = alloc1float(np);

	zero1complex(gka,nka);
	zero1complex(hp,np);
	zero1float(kp,np);

	/* scale g(x) by x sampling interval dx */
	for (ix=0; ix<nx; ++ix,x+=dx) {
		gx[ix].r = dx*g[ix].r;
		gx[ix].i = dx*g[ix].i;
	}
	/* pad g(x) with zeros */
	for (ix=nx; ix<nxfft; ++ix)
		gx[ix].r = gx[ix].i = 0.0;

	/* negate every other sample so k-axis will be centered */
	for (ix=1; ix<nx; ix+=2) {
		gx[ix].r = -gx[ix].r;
		gx[ix].i = -gx[ix].i;
	}

	/* Fourier transform g(x) to g(k) */
	pfacc(-1,nxfft,gx);

	/* wrap-around g(k) to avoid interpolation end effects */
	for (ik=0; ik<lwrap; ++ik)
		gka[ik] = gk[ik+nk-lwrap];
	for (ik=lwrap+nk; ik<lwrap+nk+lwrap; ++ik)
		gka[ik] = gk[ik-lwrap-nk];

	/* phase shift to account for non-centered x-axis */
	xshift = 0.5*(nx-1)*dx;
	for (ik=0,k=fka; ik<nka; ++ik,k+=dk) {
		phase = k*xshift;
		c = cos(phase);
		s = sin(phase);
		temp = gka[ik].r*c-gka[ik].i*s;
		gka[ik].i = gka[ik].r*s+gka[ik].i*c;
		gka[ik].r = temp;
	}

	/* compute k values at which to interpolate g(k) */
	for (ip=0,p=fp; ip<np; ++ip,p+=dp) {
		kp[ip] = w*p;

		/* if outside Nyquist bounds, do not interpolate */
		if (kp[ip]<fk && kp[ip]<ek)
			kp[ip] = fk-1000.0*dk;
		else if (kp[ip]>fk && kp[ip]>ek)
			kp[ip] = ek+1000.0*dk;
	}

	/* interpolate g(k) to obtain h(p) */
	ints8c(nka,dk,fka,gka,czero,czero,np,kp,hp);

	/* phase shift to account for non-centered x-axis and non-zero fx */
	xshift = -fx-0.5*(nx-1)*dx;
	for (ip=0; ip<np; ++ip) {
		phase = kp[ip]*xshift;
		c = cos(phase);
		s = sin(phase);
		h[ip].r = hp[ip].r*c-hp[ip].i*s;
		h[ip].i = hp[ip].r*s+hp[ip].i*c;
	}

	/* free workspace */
	free1complex(gka);
	free1complex(hp);
	free1float(kp);
}

static BeamData* beamData (int npx, float wmin, int nt, float dt, float ft, float **f)
/*****************************************************************************
Compute filtered complex beam data as a function of real and imaginary time.
******************************************************************************
      nput:
wmin		minimum frequency (in radians per unit time)
nt		number of time samples
dt		time sampling interval
ft		first time sample
f		array[nt] containing data to be filtered

Returned:	pointer to beam data
*****************************************************************************/
{
	int ntpad,ntfft,nw,iwnyq,ntrfft,ntr,nti,nwr,it,itr,iti,iw,ipx;
	float dw,fw,dtr,ftr,dti,fti,w,ti,scale,*fa;
	complex *ca,*cb,*cfi,***cf;
	BeamData *bd;
	int nf1,nf2,nf3,nf4;
	float ffw,tmpp;

	/* pad to avoid wraparound in Hilbert transform */
	ntpad = 25;

	/* fft sampling */
	ntfft = npfaro(nt+ntpad,2*(nt+ntpad));
	nw = ntfft/2+1;
	dw = 2.0*PI/(ntfft*dt);
	fw = 0.0;
	iwnyq = nw-1;

	/* filter parameters*/
	ffw=2.0*PI*1;
	nf1=ffw/dw+0.5;
	ffw=2.0*PI*10;
	nf2=ffw/dw+0.5;
	ffw=2.0*PI*35;
	nf3=ffw/dw+0.5;
	ffw=2.0*PI*45;
	nf4=ffw/dw+0.5;

	/* real time sampling (oversample for future linear interpolation) */
	ntrfft = nwr = npfao(NOVERSAMPLE*ntfft,NOVERSAMPLE*ntfft+ntfft);
	dtr = dt*ntfft/ntrfft;
	ftr = ft;
	ntr = (1+(nt+ntpad-1)*dt/dtr);
//	printf("ntr=%d,dtr=%d,ftr=%f\n",ntr,ntrfft,ftr+(ntr-1)*dtr);

	/* imaginary time sampling (exponential decay filters) */
	nti = NFILTER;
	dti = EXPMIN/(wmin*(nti-1));
	fti = 0.0;

	/* allocate space for filtered data */
	cf = alloc3complex(ntr,nti,npx);
	for(ipx=0;ipx<npx;ipx++)
	{
	/* allocate workspace */
  	 fa = alloc1float(ntfft);
	 ca = alloc1complex(nw);
	 cb = alloc1complex(ntrfft);

	 zero1float(fa,ntfft);
     zero1complex(ca,nw);
     zero1complex(cb,ntrfft);

	/* pad data with zeros */
	 for (it=0; it<nt; ++it)
		fa[it] = f[ipx][it];
	 for (it=nt; it<ntfft; ++it)
		fa[it] = 0.0;

	/* Fourier transform and scale to make complex analytic signal */
	 pfarc(1,ntfft,fa,ca);
	 for (iw=1; iw<iwnyq; ++iw) {
		ca[iw].r *= 2.0;
		ca[iw].i *= 2.0;
	 }

 	     for(iw=0;iw<nw;iw++) {

		  if(iw<nf1||iw>nf4)
		    	ca[iw]=cmplx(0.0,0.0);
		  else if(iw>=nf1&&iw<=nf2)
			    tmpp=0.54+0.46*cos(PI*(iw-nf1)/(nf2-nf1)-PI),
			    ca[iw].r=ca[iw].r*tmpp,
			    ca[iw].i=ca[iw].i*tmpp;

		   else if(iw>=nf3&&iw<=nf4)
                tmpp=0.54+0.46*cos(PI*(nf3-iw)/(nf4-nf3)),
		    	ca[iw].r=ca[iw].r*tmpp,
	     		ca[iw].i=ca[iw].i*tmpp;
	       else
		    	ca[iw]=ca[iw];
	   }

	/* loop over imaginary time */
	 for (iti=0,ti=fti; iti<nti; ++iti,ti+=dti) {

		/* apply exponential decay filter */
		for (iw=0,w=fw; iw<nw; ++iw,w+=dw) {
			scale = exp(w*ti);
			cb[iw].r = (ca[iw].r)*scale;
			cb[iw].i = (ca[iw].i)*scale;
		}
		/* pad with zeros */
		for (iw=nw; iw<nwr; ++iw)
			cb[iw].r = cb[iw].i = 0.0;

		/* inverse Fourier transform and scale */
		pfacc(-1,ntrfft,cb);

		cfi = cf[ipx][iti];
		scale = 1.0/ntfft;

		for (itr=0; itr<ntr; ++itr) {
			cfi[itr].r = scale*cb[itr].r;
			cfi[itr].i = scale*cb[itr].i;

		}

	}

	/* free workspace */
	 free1float(fa);
	 free1complex(ca);
	 free1complex(cb);
	}

	/* return beam data */
	bd = (BeamData*)alloc1(1,sizeof(BeamData));
	bd->cf=cf;
	bd->ntr = ntr;
	bd->dtr = dtr;
	bd->ftr = ftr;
	bd->nti = nti;
	bd->dti = dti;
	bd->fti = fti;
//	bd->cf = cf;
	return bd;
}

static void setCell (Cells *cells, int jx, int jz)
/*****************************************************************************
Set a cell by computing its Gaussian beam complex time and amplitude.
      f the amplitude is non-zero, set neighboring cells recursively.
******************************************************************************
      nput:
cells		pointer to cells
jx		x index of the cell to set
jz		z index of the cell to set
******************************************************************************
Notes:
To reduce the amount of memory required for recursion, the actual
computation of complex time and amplitude is performed by the cellTimeAmp()
function, so that no local variables are required in this function, except
for the input arguments themselves.
*****************************************************************************/
{
	/* if cell is out of bounds, return */
	if (jx<0 || jx>=cells->mx || jz<0 || jz>=cells->mz) return;

	/* if cell is live, return */
	if (cells->cell[jx][jz].live==cells->live) return;

	/* make cell live */
	cells->cell[jx][jz].live = cells->live;
	cells->cell[jx][jz].ip=cells->ip;
	/* compute complex time and amplitude.  If amplitude is
	 * big enough, recursively set neighboring cells. */
	if (cellTimeAmp(cells,jx,jz)) {
		setCell(cells,jx+1,jz);
		setCell(cells,jx-1,jz);
		setCell(cells,jx,jz+1);
		setCell(cells,jx,jz-1);
	}

}
static void getangle(Cell **c, int mx, int mz, float dx, float dz, int live)
{
	int i,j;
	int xc,zc;  /* cell coordinates*/
	float px,pz;
	for(i=1;i<mx-1;i++)
	{
		for(j=1;j<mz-1;j++)
		{
			if(c[i][j].live==live)
			{
				if((c[i-1][j].live==live)&&(c[i+1][j].live==live))
					px=(pow(c[i+1][j].tr,2)-pow(c[i-1][j].tr,2))/(4*c[i][j].tr*dx);
				else if((c[i-1][j].live==live)&&(c[i+1][j].live!=live))
					px=(c[i][j].tr-c[i-1][j].tr)/dx;
				else if((c[i-1][j].live!=live)&&(c[i+1][j].live==live))
					px=(c[i+1][j].tr-c[i][j].tr)/dx;
				if((c[i][j-1].live==live)&&(c[i][j+1].live==live))
					pz=(pow(c[i][j+1].tr,2)-pow(c[i][j-1].tr,2))/(4*c[i][j].tr*dz);
				else if((c[i][j-1].live==live)&&(c[i][j+1].live!=live))
					pz=(c[i][j].tr-c[i][j-1].tr)/dz;
				else if((c[i][j-1].live!=live)&&(c[i][j+1].live==live))
					pz=(c[i][j+1].tr-c[i][j].tr)/dz;
				if((px>0)&&(pz>0))
					c[i][j].angle=atan(px/pz);
				else if((px>0)&&(pz<0))
					c[i][j].angle=atan(px/pz)+PI;
				else if((px<0)&&(pz>0))
					c[i][j].angle=atan(px/pz);
				else
					c[i][j].angle=-PI+atan(px/pz);
			}
		}
	}
	for(j=1;j<mz-1;j++)
	{
		if(c[0][j].live==live)
		{
			if((c[0][j-1].live==live)&&(c[0][j+1].live==live))
				pz=(pow(c[0][j+1].tr,2)-pow(c[0][j-1].tr,2))/(4*c[0][j].tr*dz);
			else if((c[0][j-1].live==live)&&(c[0][j+1].live!=live))
				pz=(c[0][j].tr-c[0][j-1].tr)/dz;
			else if((c[0][j-1].live!=live)&&(c[0][j+1].live==live))
				pz=(c[0][j+1].tr-c[0][j].tr)/dz;
			if(c[1][j].live==live)
				px=(c[1][j].tr-c[0][j].tr)/dx;
				if((px>0)&&(pz>0))
					c[0][j].angle=atan(px/pz);
				else if((px>0)&&(pz<0))
					c[0][j].angle=atan(px/pz)+PI;
				else if((px<0)&&(pz>0))
					c[0][j].angle=atan(px/pz);
				else
					c[0][j].angle=-PI+atan(px/pz);
		}
	}
	for(i=1;i<mx-1;i++)
	{
		if(c[i][0].live==live)
		{
			if((c[i-1][0].live==live)&&(c[i+1][0].live==live))
				px=(pow(c[i+1][0].tr,2)-pow(c[i-1][0].tr,2))/(4*c[i][0].tr*dx);
			else if((c[i-1][0].live==live)&&(c[i+1][0].live!=live))
				px=(c[i][0].tr-c[i-1][0].tr)/dx;
			else if((c[i-1][0].live!=live)&&(c[i+1][0].live==live))
				px=(c[i+1][0].tr-c[i][0].tr)/dx;
			if(c[i][1].live==live)
				px=(c[i][1].tr-c[i][0].tr)/dz;
				if((px>0)&&(pz>0))
					c[i][0].angle=atan(px/pz);
				else if((px>0)&&(pz<0))
					c[i][0].angle=atan(px/pz)+PI;
				else if((px<0)&&(pz>0))
					c[i][0].angle=atan(px/pz);
				else
					c[i][0].angle=-PI+atan(px/pz);
		}
	}
}







static int cellTimeAmp (Cells *cells, int jx, int jz)
/*****************************************************************************
Compute complex and time and amplitude for a cell
******************************************************************************
      nput:
cells		pointer to cells
jx		x index of the cell to set
jz		z index of the cell to set

Returned:	1 if Gaussian amplitude is significant, 0 otherwise
*****************************************************************************/
{
	/*if you are interested in this code, please contact me*/
}










void scanimg(Cell ***c1, Cell ***c2, int live, int dead, int nt, float dt, float ft,
	float fmin, int lx, int lz, int nx, int nz, int mx, int mz, int npx, float **f,
	float**g, float ***adcig, float fpx, float dpx)
/************************************************************************************
Scan over offset ray parameters then image with beams.
*************************************************************************************
      nput parameters:
************************************************************************************/
{
	int pm,ips,ph,ipr,imx,imz,tp;
	float wmin,tmin,tempt;
	Cell **fncell;
	BeamData *bd;
	wmin=2.0*PI*fmin;
	pm=2*npx-1;

	bd=beamData(npx,wmin,nt,dt,ft,f);


/* for each midpoint ray parameter*/
	for(ips=0;ips<npx;ips++)
	{
		for(ipr=0;ipr<npx;ipr++)
		{

	    	fncell=(Cell**)alloc2(mz,mx,sizeof(Cell));
			for(imx=0;imx<mx;imx++)
			 for(imz=0;imz<mz;imz++)
               {

				fncell[imx][imz].live=0;
				fncell[imx][imz].ip=0;
				fncell[imx][imz].tr=0.0;
				fncell[imx][imz].ti=0.0;
				fncell[imx][imz].ar=0.0;
				fncell[imx][imz].ai=0.0;
				fncell[imx][imz].angle=0.0;
			  }


			for(imx=0;imx<mx;imx++)
			{
				for(imz=0;imz<mz;imz++)
				{
					tmin=1000.0;
					tp=1000;


				   if((c1[ips][imx][imz].live==live)&&(c2[ipr][imx][imz].live==live))
				   {
					  tempt=c1[ips][imx][imz].ti+c2[ipr][imx][imz].ti;
						 // printf("%d\n",tp);

				    	if((fabs(tempt)<5.0))
						{
							tmin=fabs(tempt),
							tp=ips;
						}
					}
					if((tp<1000))
					{

			    	fncell[imx][imz].ai=(c1[ips][imx][imz].ai+c2[ipr][imx][imz].ai);
			    	fncell[imx][imz].ar=(c1[ips][imx][imz].ar+c2[ipr][imx][imz].ar);
				    fncell[imx][imz].ti=-tmin;
				    fncell[imx][imz].tr=c1[ips][imx][imz].tr+c2[ipr][imx][imz].tr;
		     		fncell[imx][imz].live=live;
		    		fncell[imx][imz].ip=c2[ipr][imx][imz].ip;

					}

				}
			}
		for(imx=0;imx<mx-1;imx++)
		{
			for(imz=0;imz<mz-1;imz++)
			{
				if((fncell[imx][imz].live==live)&&(fncell[imx+1][imz].live==live)&&
					(fncell[imx][imz+1].live==live)&&(fncell[imx+1][imz+1].live==live)
                                          &&(abs(fncell[imx+1][imz+1].ip-fncell[imx][imz+1].ip)<2)
                                          &&(abs(fncell[imx][imz+1].ip-fncell[imx][imz].ip)<2)
                                          &&(abs(fncell[imx][imz].ip-fncell[imx+1][imz].ip)<2)
                                          &&(abs(fncell[imx+1][imz].ip-fncell[imx+1][imz+1].ip)<2))


					  cellBeam(fncell,c1,c2,g,imx,imz,bd,lx,lz,nx,nz,npx,ips,ipr,adcig,fpx,dpx);

			}
		}

        free2((void**)fncell);
		}
///------------------------------------------------------------------------------

	}
	free3complex(bd->cf);

}






static void cellBeam (Cell **cell, Cell ***c1, Cell ***c2, float **g, int jx, int jz, BeamData *bd, int lx, int lz,
					  int nx, int nz, int npx, int ips, int ipr, float ***adcig, float fpx, float dpx)
/*****************************************************************************
Accumulate Gaussian beam for one cell.
******************************************************************************
      nput:
cells		pointer to cells
jx		x index of the cell in which to accumulate beam
jz		z index of the cell in which to accumulate beam
*****************************************************************************/
{
	int ntr=bd->ntr,nti=bd->nti;
	float dtr=bd->dtr,ftr=bd->ftr,dti=bd->dti,fti=bd->fti;
	complex ***cf=bd->cf;
	int kxlo,kxhi,kzlo,kzhi,kx,kz,itr,iti,np;
	float ta00r,ta01r,ta10r,ta11r,ta00i,ta01i,ta10i,ta11i,
		aa00r,aa01r,aa10r,aa11r,aa00i,aa01i,aa10i,aa11i,
		tax0r,tax1r,tax0i,tax1i,aax0r,aax1r,aax0i,aax1i,
		taxzr,taxzi,aaxzr,aaxzi,
		dtax0r,dtax0i,dtax1r,dtax1i,daax0r,daax0i,daax1r,daax1i,
		dtaxzr,dtaxzi,daaxzr,daaxzi;

	float tb00r,tb01r,tb10r,tb11r,tb00i,tb01i,tb10i,tb11i,
		ab00r,ab01r,ab10r,ab11r,ab00i,ab01i,ab10i,ab11i,
		tbx0r,tbx1r,tbx0i,tbx1i,abx0r,abx1r,abx0i,abx1i,
		tbxzr,tbxzi,abxzr,abxzi,
		dtbx0r,dtbx0i,dtbx1r,dtbx1i,dabx0r,dabx0i,dabx1r,dabx1i,
		dtbxzr,dtbxzi,dabxzr,dabxzi;

	float odtr,odti,xdelta,zdelta,trn,tin,trfrac,mtrfrac,tifrac,mtifrac,
		cf0r,cf0i,cf1r,cf1i,cfr,cfi;
	float a00g,a01g,a10g,a11g;
	float b00g,b01g,b10g,b11g;
	float da0g,da1g,a0g,a1g,azg,dazg;
	float db0g,db1g,b0g,b1g,bzg,dbzg;  // a means source, b denote beam center...
	complex *cf0,*cf1;
	int iangle;

	/* inverse of time sampling intervals */
//	ftr=ftr-0.9;
	odtr = 1.0/dtr;
	odti = 1.0/dti;


	/* complex time and amplitude for each corner */
	ta00r = c1[ips][jx][jz].tr;
	ta01r = c1[ips][jx][jz+1].tr;
	ta10r = c1[ips][jx+1][jz].tr;
	ta11r = c1[ips][jx+1][jz+1].tr;
	ta00i = c1[ips][jx][jz].ti;
	ta01i = c1[ips][jx][jz+1].ti;
	ta10i = c1[ips][jx+1][jz].ti;
	ta11i = c1[ips][jx+1][jz+1].ti;
	aa00r = c1[ips][jx][jz].ar;
	aa01r = c1[ips][jx][jz+1].ar;
	aa10r = c1[ips][jx+1][jz].ar;
	aa11r = c1[ips][jx+1][jz+1].ar;
	aa00i = c1[ips][jx][jz].ai;
	aa01i = c1[ips][jx][jz+1].ai;
	aa10i = c1[ips][jx+1][jz].ai;
	aa11i = c1[ips][jx+1][jz+1].ai;

	/* complex time and amplitude for each corner */
	tb00r = c2[ipr][jx][jz].tr;
	tb01r = c2[ipr][jx][jz+1].tr;
	tb10r = c2[ipr][jx+1][jz].tr;
	tb11r = c2[ipr][jx+1][jz+1].tr;
	tb00i = c2[ipr][jx][jz].ti;
	tb01i = c2[ipr][jx][jz+1].ti;
	tb10i = c2[ipr][jx+1][jz].ti;
	tb11i = c2[ipr][jx+1][jz+1].ti;
	ab00r = c2[ipr][jx][jz].ar;
	ab01r = c2[ipr][jx][jz+1].ar;
	ab10r = c2[ipr][jx+1][jz].ar;
	ab11r = c2[ipr][jx+1][jz+1].ar;
	ab00i = c2[ipr][jx][jz].ai;
	ab01i = c2[ipr][jx][jz+1].ai;
	ab10i = c2[ipr][jx+1][jz].ai;
	ab11i = c2[ipr][jx+1][jz+1].ai;

	a00g=c1[ips][jx][jz].angle;
	a01g=c1[ips][jx][jz+1].angle;
	a10g=c1[ips][jx+1][jz].angle;
	a11g=c1[ips][jx+1][jz+1].angle;
	b00g=c2[ipr][jx][jz].angle;
	b01g=c2[ipr][jx][jz+1].angle;
	b10g=c2[ipr][jx+1][jz].angle;
	b11g=c2[ipr][jx+1][jz+1].angle;


	/* x and z samples for cell */
	kxlo = jx*lx;
	kxhi = kxlo+lx;
	if (kxhi>nx) kxhi = nx;
	kzlo = jz*lz;
	kzhi = kzlo+lz;
	if (kzhi>nz) kzhi = nz;

	/* fractional increments for linear interpolation */
	xdelta = 1.0/lx;
	zdelta = 1.0/lz;

	/* increments for times and amplitudes at top and bottom of cell */
	dtax0r = (ta10r-ta00r)*xdelta;
	dtax1r = (ta11r-ta01r)*xdelta;
	dtax0i = (ta10i-ta00i)*xdelta;
	dtax1i = (ta11i-ta01i)*xdelta;
	daax0r = (aa10r-aa00r)*xdelta;
	daax1r = (aa11r-aa01r)*xdelta;
	daax0i = (aa10i-aa00i)*xdelta;
	daax1i = (aa11i-aa01i)*xdelta;

	dtbx0r = (tb10r-tb00r)*xdelta;
	dtbx1r = (tb11r-tb01r)*xdelta;
	dtbx0i = (tb10i-tb00i)*xdelta;
	dtbx1i = (tb11i-tb01i)*xdelta;
	dabx0r = (ab10r-ab00r)*xdelta;
	dabx1r = (ab11r-ab01r)*xdelta;
	dabx0i = (ab10i-ab00i)*xdelta;
	dabx1i = (ab11i-ab01i)*xdelta;

	da0g=(a10g-a00g)*xdelta;//
	da1g=(a11g-a01g)*xdelta;//
	db0g=(b10g-b00g)*xdelta;
	db1g=(b11g-b01g)*xdelta;

	/* times and amplitudes at top-left and bottom-left of cell */
	tax0r = ta00r;
	tax1r = ta01r;
	tax0i = ta00i;
	tax1i = ta01i;
	aax0r = aa00r;
	aax1r = aa01r;
	aax0i = aa00i;
	aax1i = aa01i;

	tbx0r = tb00r;
	tbx1r = tb01r;
	tbx0i = tb00i;
	tbx1i = tb01i;
	abx0r = ab00r;
	abx1r = ab01r;
	abx0i = ab00i;
	abx1i = ab01i;

	a0g=a00g;
	a1g=a01g;
	b0g=b00g;
	b1g=b01g;


	/* loop over x samples */
	for (kx=kxlo; kx<kxhi; ++kx) {

		/* increments for time and amplitude */
		dtaxzr = (tax1r-tax0r)*zdelta;
		dtaxzi = (tax1i-tax0i)*zdelta;
		daaxzr = (aax1r-aax0r)*zdelta;
		daaxzi = (aax1i-aax0i)*zdelta;
		/* increments for time and amplitude */
		dtbxzr = (tbx1r-tbx0r)*zdelta;
		dtbxzi = (tbx1i-tbx0i)*zdelta;
		dabxzr = (abx1r-abx0r)*zdelta;
		dabxzi = (abx1i-abx0i)*zdelta;

		dazg=(a1g-a0g)*zdelta;
		dbzg=(b1g-b0g)*zdelta;

		/* time and amplitude at top of cell */
		taxzr = tax0r;
		taxzi = tax0i;
		aaxzr = aax0r;
		aaxzi = aax0i;
		/* time and amplitude at top of cell */
		tbxzr = tbx0r;
		tbxzi = tbx0i;
		abxzr = abx0r;
		abxzi = abx0i;


		azg=a0g;
		bzg=b0g;

		/* loop over z samples */
		for (kz=kzlo; kz<kzhi; ++kz) {

			if(1)
			{
            iangle=NINT(((azg-bzg)*90/PI/4));
           if(azg>bzg) iangle=iangle;
	      else  iangle=iangle;

		  if(iangle<25&&iangle>-25)
			{
	//		printf("%d\n",iangle);
			/* index of imaginary time */
			iti = tin = (taxzi+tbxzi-fti)*odti;
			if (iti<0 || iti>=nti-1) continue;
			if(((kx-kxlo)<=NINT(lx/2))&&((kz-kzlo)<=NINT(lz/2)))
				np=cell[jx][jz].ip;
			else if(((kx-kxlo)<=NINT(lx/2))&&((kz-kzlo)>NINT(lz/2)))
				np=cell[jx][jz+1].ip;
			else if(((kx-kxlo)>NINT(lx/2))&&((kz-kzlo)<=NINT(lz/2)))
				np=cell[jx+1][jz].ip;
			else if(((kx-kxlo)>NINT(lx/2))&&((kz-kzlo)>NINT(lz/2)))
				np=cell[jx+1][jz+1].ip;



			/* pointers to left and right imaginary time samples */
			cf0 = cf[np][iti];
			cf1 = cf[np][iti+1];


			/* imaginary time linear interpolation coefficients */
			tifrac = tin-iti;
			mtifrac = 1.0-tifrac;

			/* index of real time */
			itr = trn = (taxzr+tbxzr-ftr)*odtr;
			if (itr<0 || itr>=ntr-1) continue;

			/* real time linear interpolation coefficients */
			trfrac = trn-itr;
			mtrfrac = 1.0-trfrac;

			/* real and imaginary parts of complex beam data */
			cf0r = mtrfrac*cf0[itr].r+trfrac*cf0[itr+1].r;
			cf1r = mtrfrac*cf1[itr].r+trfrac*cf1[itr+1].r;
			cfr = mtifrac*cf0r+tifrac*cf1r;
			cf0i = mtrfrac*cf0[itr].i+trfrac*cf0[itr+1].i;
			cf1i = mtrfrac*cf1[itr].i+trfrac*cf1[itr+1].i;
			cfi = mtifrac*cf0i+tifrac*cf1i;
//			printf("%d\n",iangle);

			/* accumulate beam */
//			g[kx][kz] += axzr*cfr-axzi*cfi;
//			g[kx][kz] +=1;

			adcig[kx][25+iangle][kz]+= (aaxzr*cfr-aaxzi*cfi);
			}
			}

			/* increment time and amplitude */
			taxzr += dtaxzr;
			taxzi += dtaxzi;
			aaxzr += daaxzr;
			aaxzi += daaxzi;
			/* increment time and amplitude */
			tbxzr += dtbxzr;
			tbxzi += dtbxzi;
			abxzr += dabxzr;
			abxzi += dabxzi;

			azg+=dazg;
			bzg+=dbzg;
		}

		/* increment times and amplitudes at top and bottom of cell */
		tax0r += dtax0r;
		tax1r += dtax1r;
		tax0i += dtax0i;
		tax1i += dtax1i;
		aax0r += daax0r;
		aax1r += daax1r;
		aax0i += daax0i;
		aax1i += daax1i;
		/* increment times and amplitudes at top and bottom of cell */
		tbx0r += dtbx0r;
		tbx1r += dtbx1r;
		tbx0i += dtbx0i;
		tbx1i += dtbx1i;
		abx0r += dabx0r;
		abx1r += dabx1r;
		abx0i += dabx0i;
		abx1i += dabx1i;

		a0g+=da0g;
		a1g+=da1g;
		b0g+=db0g;
		b1g+=db1g;
	}

}




void zero1int(int *p,int n1)
    {int i;
     for(i=0;i<n1;i++)
        p[i]=0;
    }

void zero1float(float *p,int n1)
    {int i;
     for(i=0;i<n1;i++)
        p[i]=0.0;
    }
void zero1complex(complex *p,int n1)
    {int i;
     for(i=0;i<n1;i++)
        p[i]=cmplx(0.0,0.0);
    }

void zero2int(int **p,int n1,int n2)
    {int i,j;
     for(i=0;i<n2;i++)
        for(j=0;j<n1;j++)
        p[i][j]=0.0;
    }

void zero2float(float **p,int n1,int n2)
    {int i,j;
     for(i=0;i<n2;i++)
        for(j=0;j<n1;j++)
        p[i][j]=0.0;
    }
void zero2complex(complex **p,int n1,int n2)
    {int i,j;
     for(i=0;i<n2;i++)
        for(j=0;j<n1;j++)
        p[i][j]=cmplx(0.0,0.0);
    }

void zero3float(float ***p,int n1,int n2, int n3)
    {int i,j,k;
     for(i=0;i<n3;i++)
        for(j=0;j<n2;j++)
            for(k=0;k<n1;k++)
                p[i][j][k]=0.0;
    }

void zero3int(int ***p,int n1,int n2, int n3)
    {int i,j,k;
     for(i=0;i<n3;i++)
        for(j=0;j<n2;j++)
            for(k=0;k<n1;k++)
                p[i][j][k]=0;
    }


 void zero3complex(complex ***p,int n1,int n2, int n3)
    {int i,j,k;
     for(i=0;i<n3;i++)
        for(j=0;j<n2;j++)
            for(k=0;k<n1;k++)
                p[i][j][k]=cmplx(0.0,0.0);
    }


